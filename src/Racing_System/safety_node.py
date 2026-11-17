"""
Safety node: LIDAR-only automatic emergency braking (AEB).

On every LIDAR scan the node compares two times:
    - time to collision (TTC): how long until the car touches something if it keeps driving
      the way it is now
    - time to stop: how long the car needs to come to a halt

The closer the TTC gets to the time to stop, the harder the node reacts:
    stage 1, caution:  TTC below caution_ttc_factor x time to stop  ->  cap the speed a little
    stage 2, slow:     TTC below slow_ttc_factor x time to stop     ->  cap the speed a lot
    stage 3, stop:     TTC below the time to stop                   ->  report unsafe, car stops
Slowing early is what lets the car weave past an obstacle without ever needing stage 3.

It works that out in four steps:
    1. time_to_stop: how much warning the car needs at a given speed (reaction time + braking)
    2. time_to_collision: slide the car's outline forward along the arc it is steering until
       a LIDAR point ends up inside it, and work out how long the car takes to get there
    3. assess: compare the TTC with the time to stop and pick the stage
    4. publish_verdict: report safe/unsafe on /safety_status and the speed cap on
       /safety_speed_limit

master_control_node stops the car whenever /safety_status is False, and otherwise never lets
the follower drive faster than /safety_speed_limit.

The node also refuses to report "safe" when there are a scan with too few usable beams, or odometry that 
has stopped arriving.
"""

# Group B6

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class SafetyNode(Node):
    """
    ROS 2 node that slows, then stops, the car as the time to collision along its path runs out.

    Subscribes to /scan, the odometry topic, and the follower's drive topic.
    Publishes a safe/unsafe flag to /safety_status and a speed cap to /safety_speed_limit.
    """

    def __init__(self) -> None:
        """
        Read the parameters, set the initial state, and create the publishers and subscriptions.

        Returns:
            None

        Raises:
            ValueError: If wheelbase, brake_decel or path_step is not greater than 0, or the
            stage factors are out of order.

        Side effects:
            Declares params and creates the publishers and subscriptions.
        """
        super().__init__('safety_node')

        # values come from params.yaml, these defaults only get used if a value is missing there
        odom_topic = self.declare_parameter('odom_topic', '/odom').value
        follower_topic = self.declare_parameter('follower_topic', '/gap_follow_drive').value

        # car outline measured from the LIDAR, in metres (x forward, y left)
        self.car_front = self.declare_parameter('car_front', 0.16).value
        self.car_rear = self.declare_parameter('car_rear', 0.29).value
        self.car_half_width = self.declare_parameter('car_half_width', 0.145).value

        # steering geometry: the car turns about a point on its rear axle line
        self.lidar_to_rear_axle = self.declare_parameter('lidar_to_rear_axle', 0.23).value
        self.wheelbase = self.declare_parameter('wheelbase', 0.33).value
        self.max_steering_angle = self.declare_parameter('max_steering_angle', 0.40).value

        # braking model: time to stop = reaction_time + speed / (2 * brake_decel)
        self.reaction_time = self.declare_parameter('reaction_time', 0.15).value
        self.brake_decel = self.declare_parameter('brake_decel', 2.5).value

        # the stages before a full stop. Each one starts when the TTC drops below its ttc_factor
        # times the time to stop, and caps the follower's speed at its speed_factor times what
        # the follower asked for (but never below min_drive_speed)
        self.caution_ttc_factor = self.declare_parameter('caution_ttc_factor', 2.0).value
        self.caution_speed_factor = self.declare_parameter('caution_speed_factor', 0.75).value
        self.slow_ttc_factor = self.declare_parameter('slow_ttc_factor', 1.5).value
        self.slow_speed_factor = self.declare_parameter('slow_speed_factor', 0.5).value
        # slowest speed the car can actually drive (m/s): a cap is never set below this
        self.min_drive_speed = self.declare_parameter('min_drive_speed', 0.7).value

        # gap we want left over once the car has stopped: ahead of the bumper, and to each side of it
        self.stop_margin = self.declare_parameter('stop_margin', 0.10).value
        self.side_margin = self.declare_parameter('side_margin', 0.05).value

        # how finely the path is checked: the car outline is tested every path_step metres
        self.path_step = self.declare_parameter('path_step', 0.05).value

        # after a stop, the path must stay clear this long before the car may drive again
        self.release_time = self.declare_parameter('release_time', 0.3).value

        # input health checks
        self.odom_timeout = self.declare_parameter('odom_timeout', 0.2).value
        self.min_valid_fraction = self.declare_parameter('min_valid_fraction', 0.5).value

        if min(self.wheelbase, self.brake_decel, self.path_step) <= 0.0:
            raise ValueError('wheelbase, brake_decel and path_step must all be greater than 0')
        if not self.caution_ttc_factor >= self.slow_ttc_factor >= 1.0:
            raise ValueError('need caution_ttc_factor >= slow_ttc_factor >= 1.0')

        # the car outline again, this time measured from the rear axle (the point the car turns about)
        self.body_front = self.lidar_to_rear_axle + self.car_front  # rear axle to front bumper
        self.body_rear = self.lidar_to_rear_axle - self.car_rear    # rear axle to rear bumper (negative)

        # latest speed from odometry, and when it arrived (None until the first message)
        self.measured_speed = 0.0
        self.last_odom_time = None

        # latest command from the follower: what the car will do if we let it drive
        self.commanded_speed = 0.0
        self.commanded_steering = 0.0

        # when we last saw danger, and why we are currently stopped (None while safe)
        self.last_danger_time = float('-inf')
        self.stop_reason = None

        # direction of every beam, built on the first scan and reused afterwards
        self.beam_layout = None
        self.beam_cos = None
        self.beam_sin = None
        self.beam_is_forward = None

        self.status_publisher = self.create_publisher(Bool, '/safety_status', 10)
        self.speed_limit_publisher = self.create_publisher(Float32, '/safety_speed_limit', 10)

        # only ever keep the newest scan: braking on an old scan is worse than skipping one
        newest_scan_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_reading, newest_scan_only)
        self.odom_subscription = self.create_subscription(
            Odometry, odom_topic, self.odom_reading, 10)
        self.follower_subscription = self.create_subscription(
            AckermannDriveStamped, follower_topic, self.follower_reading, 10)

    def now_s(self) -> float:
        """
        Get the current time from the node clock.

        Returns:
            float: Time in seconds.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_reading(self, msg: Odometry) -> None:
        """
        Store how fast the car is moving.

        Args:
            msg (Odometry): Odometry message with the car's velocity.

        Returns:
            None

        Side effects:
            Updates measured_speed (m/s, never negative) and last_odom_time.
        """
        # the car only drives forward, and the sign of linear.x is different on the car and in
        # the sim, so use the size of the velocity and ignore its sign
        self.measured_speed = abs(msg.twist.twist.linear.x)
        self.last_odom_time = self.now_s()

    def follower_reading(self, msg: AckermannDriveStamped) -> None:
        """
        Store the speed and steering angle the follower is asking for.

        Args:
            msg (AckermannDriveStamped): The follower's latest drive command.

        Returns:
            None

        Side effects:
            Updates commanded_speed (m/s) and commanded_steering (rad, positive = left).
        """
        self.commanded_speed = max(msg.drive.speed, 0.0)
        self.commanded_steering = msg.drive.steering_angle

    def lidar_reading(self, msg: LaserScan) -> None:
        """
        Make one safety decision from one LIDAR scan and publish it.

        Args:
            msg (LaserScan): LIDAR scan with ranges and beam angles.

        Returns:
            None

        Side effects:
            Publishes to /safety_status and /safety_speed_limit.
        """
        now = self.now_s()
        danger, speed_limit = self.assess(msg, now)
        self.publish_verdict(danger, speed_limit, now)

    def assess(self, msg: LaserScan, now: float) -> tuple[str | None, float]:
        """
        Decide how the car must react to this scan: stop, slow down, or carry on.

        Args:
            msg (LaserScan): LIDAR scan with ranges and beam angles.
            now (float): Current time in seconds.

        Returns:
            tuple[str | None, float]: A short reason if the car must stop (None if it need not),
            and the fastest speed the follower may drive in m/s (infinity for no cap).
        """
        # without a fresh speed reading we can't work out a time to collision
        if self.last_odom_time is None or now - self.last_odom_time > self.odom_timeout:
            return 'no odometry', 0.0

        points = self.scan_to_points(msg)
        if points is None:
            return 'LIDAR blind (too few usable beams)', 0.0

        # three speeds matter:
        #   wanted:  what the car does if we don't step in (the follower's speed, or the
        #            measured speed if the car is somehow going faster)
        #   caution: the stage 1 cap
        #   slow:    the stage 2 cap, the slowest we would ever make the car drive
        wanted_speed = max(self.measured_speed, self.commanded_speed)
        caution_speed = self.capped_speed(self.caution_speed_factor)
        slow_speed = self.capped_speed(self.slow_speed_factor)

        # stage 3, stop: the TTC is shorter than the time the car needs to stop. Judged at the
        # slow speed (or the measured speed while the car is still faster than that), so that
        # slowing down in stages 1 and 2 really can keep the car out of stage 3. While the car
        # is stopped, the slow speed is the speed it would set off at, so it is only let go when
        # driving off again is safe (no stop/go/stop/go)
        stop_check_speed = max(self.measured_speed, slow_speed)
        time_to_stop = self.time_to_stop(stop_check_speed)
        ttc = self.time_to_collision(*points, stop_check_speed, look_ahead_s=time_to_stop)
        if ttc < time_to_stop:
            return 'obstacle in path (TTC below time to stop)', 0.0

        # stages 2 and 1: the TTC is still longer than the time to stop, but not by much.
        # Judged at the speed the car wants to go
        time_to_stop = self.time_to_stop(wanted_speed)
        slow_ttc = self.slow_ttc_factor * time_to_stop
        caution_ttc = self.caution_ttc_factor * time_to_stop
        ttc = self.time_to_collision(*points, wanted_speed, look_ahead_s=caution_ttc)
        if ttc < slow_ttc:
            return None, slow_speed
        if ttc < caution_ttc:
            return None, caution_speed

        return None, math.inf

    def capped_speed(self, speed_factor: float) -> float:
        """
        Work out the speed a stage caps the car at.

        Args:
            speed_factor (float): Share of the follower's speed the stage allows (0 to 1).

        Returns:
            float: Speed in m/s: speed_factor times the follower's speed, but not below
            min_drive_speed (the car can't drive slower) and not above what the follower asked for.
        """
        capped = max(speed_factor * self.commanded_speed, self.min_drive_speed)
        return min(capped, self.commanded_speed)

    def scan_to_points(self, msg: LaserScan) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Convert a scan into obstacle points in the LIDAR frame (x forward, y left).

        Args:
            msg (LaserScan): LIDAR scan with ranges and beam angles.

        Returns:
            tuple[np.ndarray, np.ndarray] | None: x and y of every obstacle point in metres, or
            None if too few of the forward beams returned a usable reading to trust the scan.

        Side effects:
            Rebuilds the beam direction tables if the scan layout changed.
        """
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        self.update_beam_tables(msg)

        # a usable reading is a real number inside the range the LIDAR can measure
        # (nan, inf and 0 all mean the beam measured nothing)
        has_reading = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)

        # an empty or mostly-empty scan is a broken or covered LIDAR, not an empty track:
        # on our track there is always a wall in range ahead
        forward_readings = has_reading[self.beam_is_forward]
        if forward_readings.size == 0 or forward_readings.mean() < self.min_valid_fraction:
            return None

        x = ranges[has_reading] * self.beam_cos[has_reading]
        y = ranges[has_reading] * self.beam_sin[has_reading]

        # anything inside the car's own outline is part of the car (mounts, wires), not an obstacle
        is_car = (x >= -self.car_rear) & (x <= self.car_front) & (np.abs(y) <= self.car_half_width)
        return x[~is_car], y[~is_car]

    def update_beam_tables(self, msg: LaserScan) -> None:
        """
        Work out the direction of every beam, once, and reuse it for every later scan.

        Args:
            msg (LaserScan): LIDAR scan with beam angles.

        Returns:
            None

        Side effects:
            Sets beam_layout, beam_cos, beam_sin and beam_is_forward when the scan layout changes
            (in practice only on the first scan).
        """
        layout = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if layout == self.beam_layout:
            return

        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        self.beam_layout = layout
        self.beam_cos = np.cos(angles)
        self.beam_sin = np.sin(angles)
        # the beams pointing ahead of the car, used to judge whether the scan can be trusted
        self.beam_is_forward = np.abs(angles) <= math.pi / 2.0

    def time_to_stop(self, speed: float) -> float:
        """
        Work out how much warning the car needs to stop from a given speed.

        This is the TTC threshold. The car covers speed * reaction_time before the brakes bite,
        then speed^2 / (2 * brake_decel) while braking. Dividing that stopping distance by the
        speed turns it into a time, so it can be compared with a TTC:

            time to stop = reaction_time + speed / (2 * brake_decel)

        Args:
            speed (float): Speed in m/s.

        Returns:
            float: Time in seconds. An obstacle with a TTC below this can't be avoided by braking.
        """
        return self.reaction_time + speed / (2.0 * self.brake_decel)

    def time_to_collision(self, x: np.ndarray, y: np.ndarray, speed: float,
                          look_ahead_s: float) -> float:
        """
        Work out how long until the car touches an obstacle point, driving as it is now.

        The car's outline is placed where it will be every path_step metres along the arc it is
        steering. At each placement a point counts as a hit if it is:
            - inside the car body itself, or
            - inside the bumper zone: the strip stop_margin deep in front of the bumper,
              side_margin wider than the car on each side.
        The TTC is the time the car takes, at the given speed, to reach the first placement
        with a hit.

        Only the bumper zone is padded. Padding the sides of the body would stop the car for a
        wall it is merely driving alongside.

        Args:
            x (np.ndarray): Forward position of each obstacle point from the LIDAR, in metres.
            y (np.ndarray): Leftward position of each obstacle point from the LIDAR, in metres.
            speed (float): Speed to assume the car keeps, in m/s.
            look_ahead_s (float): How far into the future to check, in seconds.

        Returns:
            float: Time to collision in seconds. 0.0 if something is already in the bumper zone
            (even when standing still), and infinity if nothing is hit within look_ahead_s.
        """
        if x.size == 0:
            return math.inf

        # 1. how far along the path each placement is: 0, path_step, 2 * path_step, ...
        #    out to where the car will be look_ahead_s from now
        look_ahead_m = speed * look_ahead_s
        travel = np.arange(0.0, look_ahead_m + self.path_step, self.path_step)

        # 2. where the rear axle is after each of those distances (bicycle model).
        #    steering is limited to what the servo can do, so a wild command can't bend the path
        steering = float(np.clip(
            self.commanded_steering, -self.max_steering_angle, self.max_steering_angle))
        curvature = math.tan(steering) / self.wheelbase  # 1 / turn radius, positive = left

        if abs(curvature) < 1e-6:
            # driving straight
            heading = np.zeros_like(travel)
            moved_forward = travel
            moved_left = np.zeros_like(travel)
        else:
            # driving along a circle
            heading = curvature * travel
            moved_forward = np.sin(heading) / curvature
            moved_left = (1.0 - np.cos(heading)) / curvature

        # 3. see every point from every placement of the car.
        #    rows = placements along the path, columns = obstacle points
        offset_x = (x + self.lidar_to_rear_axle)[None, :] - moved_forward[:, None]
        offset_y = y[None, :] - moved_left[:, None]
        cos_heading = np.cos(heading)[:, None]
        sin_heading = np.sin(heading)[:, None]
        ahead = cos_heading * offset_x + sin_heading * offset_y       # ahead of the rear axle
        sideways = np.abs(cos_heading * offset_y - sin_heading * offset_x)  # from the centre line

        # 4. is any point inside the body or the bumper zone at each placement?
        in_body = ((ahead >= self.body_rear) & (ahead <= self.body_front)
                   & (sideways <= self.car_half_width))
        in_bumper_zone = ((ahead > self.body_front)
                          & (ahead <= self.body_front + self.stop_margin)
                          & (sideways <= self.car_half_width + self.side_margin))

        placement_hit = np.any(in_body | in_bumper_zone, axis=1)
        if not placement_hit.any():
            return math.inf

        # 5. TTC = time to reach the first placement with a hit
        distance_to_hit = float(travel[np.argmax(placement_hit)])
        if distance_to_hit == 0.0:
            return 0.0  # already inside the bumper zone, even when standing still
        return distance_to_hit / speed

    def publish_verdict(self, danger: str | None, speed_limit: float, now: float) -> None:
        """
        Publish safe/unsafe and the speed cap.

        After any danger the node keeps reporting unsafe until the path has stayed clear for
        release_time seconds, so one noisy scan can't release the car.

        Args:
            danger (str | None): Reason the car must stop, or None if it need not.
            speed_limit (float): Fastest speed the follower may drive, in m/s (infinity = no cap).
            now (float): Current time in seconds.

        Returns:
            None

        Side effects:
            Publishes to /safety_speed_limit and /safety_status, updates last_danger_time and
            stop_reason, and logs once when the car is stopped or released (never once per scan,
            and never for the slowing stages).
        """
        if danger is not None:
            self.last_danger_time = now
            if danger != self.stop_reason:
                self.get_logger().warn(f'Safety stop: {danger}')
                self.stop_reason = danger

        is_safe = danger is None and now - self.last_danger_time >= self.release_time

        if is_safe and self.stop_reason is not None:
            self.get_logger().info('Safety released: path clear')
            self.stop_reason = None

        # the cap goes out first, so it is already in place when master_control hears "safe"
        self.speed_limit_publisher.publish(Float32(data=speed_limit if is_safe else 0.0))
        self.status_publisher.publish(Bool(data=is_safe))


def main(args: list | None = None) -> None:
    """
    Start the safety node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit (including Ctrl+C).
    """
    rclpy.init(args=args)
    safety_node = SafetyNode()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
