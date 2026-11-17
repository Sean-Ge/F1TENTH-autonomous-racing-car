"""
Gap follow node: drives the car toward the widest open gap in the LIDAR scan.

Every scan goes through the same steps:
    1. scan_to_points: turn the LIDAR readings into (x, y) obstacle points
    2. cast_rays: grow every point into a disc (half the car's width plus some clearance) and
       fan rays out from the LIDAR, one per degree. Each ray's reach is how far it gets before
       entering a disc, so a long ray is a direction the whole car fits down
    3. choose_gap: group the longest rays into gaps and aim at the middle of the widest one.
       If no direction has room with the full clearance (a narrow passage), steps 2 and 3 are
       repeated with a much smaller clearance, so the car squeezes through instead of stopping
    4. steer_toward: turn that aim point into a steering angle (pure pursuit)
    5. choose_speed: go as fast as the room ahead and the tightness of the turn allow
    6. publish_drive: send the command on /gap_follow_drive

If the scan can't be trusted, or no ray is long enough to drive along, the node commands a stop.
"""

# Group B6

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


class GapFollow(Node):
    """
    ROS 2 node that casts rays through the LIDAR scan and drives into the widest open gap.

    Subscribes to /scan and publishes steering and speed commands to /gap_follow_drive.
    """

    def __init__(self) -> None:
        """
        Read the parameters, build the ray fan, and create the subscription and publisher.

        Returns:
            None

        Raises:
            ValueError: If a parameter that must be greater than 0 is not.

        Side effects:
            Declares params, builds the ray fan, and creates the /scan subscription and the
            /gap_follow_drive publisher.
        """
        super().__init__('gap_follow_node')

        # values come from params.yaml, these defaults only get used if a value is missing there

        # car geometry (same measurements as safety_node)
        self.car_half_width_m = float(self.declare_parameter('car_half_width_m', 0.145).value)
        self.wheelbase_m = float(self.declare_parameter('wheelbase_m', 0.33).value)
        self.lidar_to_rear_axle_m = float(
            self.declare_parameter('lidar_to_rear_axle_m', 0.23).value)

        # ray casting: how much room to keep, how far to look, and how wide the ray fan is
        self.wall_clearance_m = float(self.declare_parameter('wall_clearance_m', 0.10).value)
        self.tight_clearance_m = float(self.declare_parameter('tight_clearance_m', 0.03).value)
        self.max_range_m = float(self.declare_parameter('max_range_m', 2.2).value)
        self.ray_half_fov_deg = int(self.declare_parameter('ray_half_fov_deg', 90).value)
        self.beams_per_group = int(self.declare_parameter('beams_per_group', 4).value)
        self.min_valid_fraction = float(self.declare_parameter('min_valid_fraction', 0.5).value)

        # choosing a gap
        self.gap_depth_ratio = float(self.declare_parameter('gap_depth_ratio', 0.9).value)
        self.gap_switch_ratio = float(self.declare_parameter('gap_switch_ratio', 1.3).value)
        self.min_valid_reach_m = float(self.declare_parameter('min_valid_reach_m', 0.3).value)

        # steering
        self.lookahead_m = float(self.declare_parameter('lookahead_m', 1.0).value)
        self.max_steering_rad = float(self.declare_parameter('max_steering_rad', 0.40).value)
        self.max_steering_rate_rad_s = math.radians(
            float(self.declare_parameter('max_steering_rate_dps', 200.0).value))
        self.steering_offset_rad = float(self.declare_parameter('steering_offset_rad', 0.0).value)

        # speed
        self.max_speed_mps = float(self.declare_parameter('max_speed_mps', 1.4).value)
        self.min_speed_mps = float(self.declare_parameter('min_speed_mps', 1.0).value)
        self.slow_decel_mps2 = float(self.declare_parameter('slow_decel_mps2', 1.5).value)
        self.stop_gap_m = float(self.declare_parameter('stop_gap_m', 0.3).value)
        self.max_lateral_accel_mps2 = float(
            self.declare_parameter('max_lateral_accel_mps2', 3.0).value)

        must_be_positive = {
            'wheelbase_m': self.wheelbase_m,
            'max_range_m': self.max_range_m,
            'beams_per_group': self.beams_per_group,
            'lookahead_m': self.lookahead_m,
            'slow_decel_mps2': self.slow_decel_mps2,
            'max_lateral_accel_mps2': self.max_lateral_accel_mps2,
        }
        for name, value in must_be_positive.items():
            if value <= 0:
                raise ValueError(f'{name} must be greater than 0, got {value}')

        # every obstacle point is grown into a disc this big. A ray that misses every disc is a
        # line the centre of the car can drive along with wall_clearance_m to spare on each side
        self.disc_radius_m = self.car_half_width_m + self.wall_clearance_m
        # the smaller disc used when nothing fits with the full clearance. The narrowest passage
        # the car will drive through is twice this radius
        self.tight_disc_radius_m = self.car_half_width_m + min(
            self.tight_clearance_m, self.wall_clearance_m)
        # the full clearance is used whenever some direction has this much room with it:
        # enough to fit the whole aim point
        self.roomy_min_reach_m = min(self.lookahead_m, self.max_range_m)

        # the ray fan: one ray per degree from -ray_half_fov_deg to +ray_half_fov_deg,
        # positive = left. The rays start at the LIDAR, so a ray's bearing is just its angle
        angles_deg = np.arange(-self.ray_half_fov_deg, self.ray_half_fov_deg + 1, dtype=np.float64)
        self.bearing_rad = np.deg2rad(angles_deg)
        # unit direction (x forward, y left) of every ray, shape (rays, 2)
        self.ray_dirs = np.stack([np.cos(self.bearing_rad), np.sin(self.bearing_rad)], axis=1)

        # direction of every group of beams, built on the first scan and reused afterwards
        self.scan_layout = None
        self.group_cos = None
        self.group_sin = None
        self.group_is_forward = None

        # where we aimed on the last scan, so we stay with the same gap from scan to scan
        self.prev_aim_bearing_rad = 0.0

        # last steering angle sent and when (node clock), for the steering rate limit
        self.prev_steering_rad = 0.0
        self.prev_steering_time_s: Optional[float] = None

        # why we are currently stopped (None while driving), so each stop is only logged once
        self.stop_reason: Optional[str] = None

        # only ever keep the newest scan: steering from an old scan is worse than skipping one
        newest_scan_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, newest_scan_only)
        # must be /gap_follow_drive so master_control forwards it in gap mode
        self.publisher = self.create_publisher(AckermannDriveStamped, '/gap_follow_drive', 10)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Run the whole pipeline on one LIDAR scan and publish a drive command.

        Args:
            msg (LaserScan): LIDAR scan with ranges and beam angles.

        Returns:
            None

        Side effects:
            Publishes to /gap_follow_drive and updates the aim, steering and stop state.
        """
        # 1. scan -> obstacle points around the car
        points = self.scan_to_points(msg)
        if points is None:
            self.publish_stop('LIDAR blind (too few usable beams)')
            return

        # 2. how far the car could drive in each direction, keeping the full clearance
        reach_m = self.cast_rays(points, self.disc_radius_m)

        # 3. where to aim. The full clearance is what we would like, not what the car needs:
        #    in a passage too narrow for it, settle for the tight clearance rather than stopping
        #    in front of a gap the car fits through
        aim = self.choose_gap(reach_m, self.roomy_min_reach_m)
        if aim is None:
            tight_reach_m = self.cast_rays(points, self.tight_disc_radius_m)
            aim = self.choose_gap(tight_reach_m, self.min_valid_reach_m)
        if aim is None:
            self.publish_stop('boxed in (no direction with room to drive)')
            return
        aim_bearing_rad, aim_distance_m = aim

        # 4. steering angle that takes the car there, turned no faster than the servo can
        steering_rad = self.steer_toward(aim_bearing_rad, aim_distance_m)
        steering_rad = self.limit_steering_rate(steering_rad)

        # 5. speed that suits the room ahead and the turn. Always judged with the full clearance,
        #    so the car drops to its slowest speed while it squeezes through a narrow passage
        speed_mps = self.choose_speed(reach_m, steering_rad)

        # 6. send it
        self.publish_drive(steering_rad, speed_mps)

    def scan_to_points(self, msg: LaserScan) -> Optional[np.ndarray]:
        """
        Convert the scan into (x, y) obstacle points in the LIDAR frame (x forward, y left).

        Neighbouring beams are almost the same direction, so they are taken beams_per_group at
        a time and only the closest reading of each group is kept. That keeps thin obstacles
        (the closest beam always survives) while giving cast_rays far fewer points to check.

        Args:
            msg (LaserScan): LIDAR scan.

        Returns:
            Optional[np.ndarray]: Obstacle points, shape (N, 2), in metres. None if too few of
            the forward beams returned a usable reading to trust the scan.

        Side effects:
            Rebuilds the beam direction tables if the scan layout changed.
        """
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        self.update_beam_tables(msg)

        # a usable reading is a real number inside the range the LIDAR can measure
        # (nan, inf and 0 all mean the beam measured nothing)
        has_reading = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)

        # closest usable reading in each group of beams (inf if the whole group measured nothing).
        # the few beams left over at the end of the scan point behind the car and are not needed
        group_count = len(self.group_cos)
        grouped_beams = group_count * self.beams_per_group
        nearest_m = np.where(has_reading, ranges, np.inf)[:grouped_beams].reshape(
            group_count, self.beams_per_group).min(axis=1)
        group_has_reading = np.isfinite(nearest_m)

        # an empty or mostly-empty scan is a broken or covered LIDAR, not an empty track:
        # on our track there is always a wall in range ahead
        forward_readings = group_has_reading[self.group_is_forward]
        if forward_readings.size == 0 or forward_readings.mean() < self.min_valid_fraction:
            return None

        # a point further than max_range_m + disc_radius_m can't touch any ray, so skip it
        keep = group_has_reading & (nearest_m <= self.max_range_m + self.disc_radius_m)
        return np.stack([nearest_m[keep] * self.group_cos[keep],
                         nearest_m[keep] * self.group_sin[keep]], axis=1)

    def update_beam_tables(self, msg: LaserScan) -> None:
        """
        Work out the direction of every group of beams, once, and reuse it for every later scan.

        Args:
            msg (LaserScan): LIDAR scan with beam angles.

        Returns:
            None

        Side effects:
            Sets scan_layout, group_cos, group_sin and group_is_forward when the scan layout
            changes (in practice only on the first scan).
        """
        layout = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if layout == self.scan_layout:
            return

        # each group points along the middle of the beams it is made of
        group_count = len(msg.ranges) // self.beams_per_group
        first_beam = np.arange(group_count) * self.beams_per_group
        middle_beam = first_beam + (self.beams_per_group - 1) / 2.0
        angles = msg.angle_min + middle_beam * msg.angle_increment

        self.scan_layout = layout
        self.group_cos = np.cos(angles)
        self.group_sin = np.sin(angles)
        # the groups pointing ahead of the car, used to judge whether the scan can be trusted
        self.group_is_forward = np.abs(angles) <= math.pi / 2.0

    def cast_rays(self, points: np.ndarray, disc_radius_m: float) -> np.ndarray:
        """
        Measure how far each ray gets before it enters any obstacle disc.

        Every obstacle point is a disc of radius disc_radius_m (half the car's width plus a
        clearance), so a free ray means the centre of the car could drive along it without its
        sides coming within that clearance of anything.

        For a ray with direction d and a point p:
            along = p . d                        how far along the ray the point is
            perp^2 = |p|^2 - along^2             squared distance from the ray to the point
        The ray hits the disc if the point is ahead (along > 0) and perp < disc_radius_m, and
        enters it at along - sqrt(disc_radius_m^2 - perp^2). If the car is already inside a disc,
        rays heading toward that point get 0 and rays heading away aren't blocked by it.

        Args:
            points (np.ndarray): Obstacle points from scan_to_points(), shape (N, 2).
            disc_radius_m (float): Radius every obstacle point is grown to, in metres.

        Returns:
            np.ndarray: Free distance (m) of each ray, capped at max_range_m.
        """
        reach_m = np.full(len(self.bearing_rad), self.max_range_m)
        if len(points) == 0:
            return reach_m

        radius_sq = disc_radius_m ** 2

        # (N, rays): every point against every ray at once
        along = points @ self.ray_dirs.T
        perp_sq = np.sum(points ** 2, axis=1)[:, None] - along ** 2
        hit = (along > 0.0) & (perp_sq < radius_sq)

        # distance along the ray to where it enters the disc (0 if it starts inside)
        entry = np.maximum(along - np.sqrt(np.maximum(radius_sq - perp_sq, 0.0)), 0.0)
        entry = np.where(hit, entry, np.inf)

        # each ray stops at the first disc it enters; rays that hit nothing are open
        return np.minimum(entry.min(axis=0), reach_m)

    def choose_gap(self, reach_m: np.ndarray,
                   min_reach_m: float) -> Optional[Tuple[float, float]]:
        """
        Pick the gap to drive into and the point in it to aim at.

        A ray is "open" if it reaches nearly as far as the longest ray (gap_depth_ratio of it).
        Open rays that sit next to each other form a gap. We aim at the middle ray of the widest
        gap, but stay with the gap we were already aiming at unless another one is clearly wider
        (gap_switch_ratio times), so the car can't dither between two similar gaps.

        Args:
            reach_m (np.ndarray): Free distance of each ray in metres.
            min_reach_m (float): The longest ray must reach at least this far, in metres, for
                these rays to be worth driving along.

        Returns:
            Optional[Tuple[float, float]]: (bearing in radians, positive = left, and distance in
            metres) of the aim point, or None if no ray reaches min_reach_m.

        Side effects:
            Updates prev_aim_bearing_rad (only when an aim point is returned).
        """
        best_reach_m = float(reach_m.max())
        if best_reach_m < min_reach_m:
            return None

        # find each run of neighbouring open rays: is_open goes False -> True at the start of a
        # gap and True -> False just after its end
        is_open = reach_m >= self.gap_depth_ratio * best_reach_m
        changes = np.diff(np.concatenate(([False], is_open, [False])).astype(np.int8))
        first_ray = np.flatnonzero(changes == 1)
        last_ray = np.flatnonzero(changes == -1) - 1

        width_rays = last_ray - first_ray + 1
        middle_ray = (first_ray + last_ray) // 2

        # the widest gap, and the gap closest to where we aimed last scan
        widest = int(np.argmax(width_rays))
        current = int(np.argmin(np.abs(self.bearing_rad[middle_ray] - self.prev_aim_bearing_rad)))
        if width_rays[widest] > self.gap_switch_ratio * width_rays[current]:
            chosen = widest
        else:
            chosen = current

        aim_ray = int(middle_ray[chosen])
        aim_bearing_rad = float(self.bearing_rad[aim_ray])
        # aim lookahead_m down the ray, or at its end if it is shorter than that
        aim_distance_m = min(float(reach_m[aim_ray]), self.lookahead_m)

        self.prev_aim_bearing_rad = aim_bearing_rad
        return aim_bearing_rad, aim_distance_m

    def steer_toward(self, aim_bearing_rad: float, aim_distance_m: float) -> float:
        """
        Work out the steering angle whose arc passes through the aim point (pure pursuit).

        The car turns about its rear axle, so the aim point is first measured from there. The
        circle that leaves the rear axle pointing straight ahead and passes through a point
        (x, y) has curvature 2y / (x^2 + y^2), and a car with wheelbase L drives that circle
        with a steering angle of atan(L * curvature).

        Steering straight at the ray's bearing instead would turn far too hard whenever the aim
        point is more than a couple of car lengths away.

        Args:
            aim_bearing_rad (float): Direction of the aim point from the LIDAR, positive = left.
            aim_distance_m (float): Distance of the aim point from the LIDAR in metres.

        Returns:
            float: Steering angle in radians, positive = left, within +-max_steering_rad.
        """
        # the aim point measured from the rear axle (x forward, y left)
        x = aim_distance_m * math.cos(aim_bearing_rad) + self.lidar_to_rear_axle_m
        y = aim_distance_m * math.sin(aim_bearing_rad)

        distance_sq = x * x + y * y
        if distance_sq < 1e-9:
            return 0.0

        curvature = 2.0 * y / distance_sq  # 1 / turn radius, positive = left
        steering_rad = math.atan(self.wheelbase_m * curvature)
        return float(np.clip(steering_rad, -self.max_steering_rad, self.max_steering_rad))

    def limit_steering_rate(self, target_rad: float) -> float:
        """
        Move the steering toward target_rad by at most max_steering_rate_dps per second.

        Uses the node clock to measure how long it has been since the last command, so the
        limit is in real degrees per second no matter how fast scans arrive. It should match
        how fast the servo can really turn the wheels: any faster and the command is asking
        for something the car can't do. A max_steering_rate_dps of 0 or less turns it off.

        Args:
            target_rad (float): Steering angle steer_toward wants, in radians, positive = left.

        Returns:
            float: Steering angle to send, in radians.

        Side effects:
            Updates prev_steering_rad and prev_steering_time_s.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # limit off, first command, or the clock jumped backwards (sim reset): no limiting
        if (self.max_steering_rate_rad_s <= 0.0 or self.prev_steering_time_s is None
                or now_s < self.prev_steering_time_s):
            limited_rad = target_rad
        else:
            # how far the wheels may turn in the time since the last command
            max_step_rad = self.max_steering_rate_rad_s * (now_s - self.prev_steering_time_s)
            step_rad = float(np.clip(target_rad - self.prev_steering_rad,
                                     -max_step_rad, max_step_rad))
            limited_rad = self.prev_steering_rad + step_rad

        self.prev_steering_rad = limited_rad
        self.prev_steering_time_s = now_s
        return limited_rad

    def choose_speed(self, reach_m: np.ndarray, steering_rad: float) -> float:
        """
        Pick the fastest speed that suits the room ahead and the tightness of the turn.

        Two limits, and the car takes the lower one:
            braking:   from this speed the car can slow at slow_decel_mps2 and still have
                       stop_gap_m left, within the free distance along the ray its front wheels
                       are pointing down.            speed^2 = 2 * decel * (reach - stop_gap)
            cornering: sideways acceleration (speed^2 / turn radius) stays under
                       max_lateral_accel_mps2.       speed^2 = lateral_accel * turn radius
        The result is kept between min_speed_mps (the car can't drive smoothly below it) and
        max_speed_mps.

        Args:
            reach_m (np.ndarray): Free distance of each ray in metres.
            steering_rad (float): Steering angle being sent, in radians, positive = left.

        Returns:
            float: Speed in m/s.
        """
        # the ray closest to the direction the front wheels point (rays are one per degree)
        wheel_ray = int(round(math.degrees(steering_rad))) + self.ray_half_fov_deg
        wheel_ray = min(max(wheel_ray, 0), len(reach_m) - 1)
        room_m = max(float(reach_m[wheel_ray]) - self.stop_gap_m, 0.0)
        braking_limit_mps = math.sqrt(2.0 * self.slow_decel_mps2 * room_m)

        curvature = abs(math.tan(steering_rad)) / self.wheelbase_m  # 1 / turn radius
        if curvature < 1e-6:
            cornering_limit_mps = math.inf  # driving straight
        else:
            cornering_limit_mps = math.sqrt(self.max_lateral_accel_mps2 / curvature)

        speed_mps = min(self.max_speed_mps, braking_limit_mps, cornering_limit_mps)
        return max(speed_mps, self.min_speed_mps)

    def publish_drive(self, steering_rad: float, speed_mps: float) -> None:
        """
        Publish a drive command on /gap_follow_drive.

        Args:
            steering_rad (float): Steering angle in radians, positive = left.
            speed_mps (float): Speed in m/s.

        Returns:
            None

        Side effects:
            Publishes an AckermannDriveStamped message to /gap_follow_drive, and logs once if
            the node had been stopped and is now driving again.
        """
        if self.stop_reason is not None and speed_mps > 0.0:
            self.get_logger().info('Gap follow driving again')
            self.stop_reason = None

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        # servo trim is added last, and the total still has to be something the servo can do
        msg.drive.steering_angle = float(np.clip(
            steering_rad + self.steering_offset_rad,
            -self.max_steering_rad, self.max_steering_rad))
        msg.drive.speed = float(speed_mps)
        self.publisher.publish(msg)

    def publish_stop(self, reason: str) -> None:
        """
        Command zero speed, keeping the wheels where they are.

        Args:
            reason (str): Why the car can't drive, for the log.

        Returns:
            None

        Side effects:
            Publishes a zero-speed command to /gap_follow_drive and logs the reason once (not
            on every scan).
        """
        if reason != self.stop_reason:
            self.get_logger().warn(f'Gap follow stopped: {reason}')
            self.stop_reason = reason

        self.publish_drive(self.prev_steering_rad, 0.0)


def main(args: Optional[list] = None) -> None:
    """
    Start the gap follow node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit (including Ctrl+C).
    """
    rclpy.init(args=args)
    gap_follow_node = GapFollow()

    try:
        rclpy.spin(gap_follow_node)
    except KeyboardInterrupt:
        pass
    finally:
        gap_follow_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
