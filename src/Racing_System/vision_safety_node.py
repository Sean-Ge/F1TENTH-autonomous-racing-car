"""
Vision-based safety node for partial-braking automatic emergency braking (AEB).

Uses the RealSense depth camera and odometry to compute time to collision (TTC)
and applies mild, hard, or full braking depending on how close a collision is.
Same braking stages and topics as the Milestone 2 LIDAR safety node, so it plugs
straight into master_control_node.
"""

# Group B6

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from std_msgs.msg import Bool # need this import for safety_node 'safe' flag
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class VisionSafetyNode(Node):
    """
    ROS 2 node that brakes the car when a collision is imminent, using the depth camera.

    Subscribes to the depth image topic, the odometry topic, and /drive. Publishes brake commands to /safety_drive and a safe/unsafe flag to /safety_status.
    """

    def __init__(self) -> None:
        """
        Set up parameters, initial state, publishers, subscriptions, and the camera watchdog timer.

        Returns:
            None

        Side effects:
            Reads params, sets speed and steering angle to 0.0, sets up publishers and subscriptions, and starts the watchdog timer
        """
        # own node name so params.yaml can hold camera-specific values separate from the LIDAR node
        super().__init__('vision_safety_node')

        # values come from params.yaml, these defaults only get used if a value is missing there
        self.depth_topic = self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw').value
        self.odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self.min_drive_speed = self.declare_parameter('min_drive_speed', 0.7).value

        # rows of the depth image to check, picked by going through our recorded bag data
        # (this band stands in for the LIDAR beams in front of the car)
        self.row_top = self.declare_parameter('row_top', 280).value
        self.row_bottom = self.declare_parameter('row_bottom', 380).value
        # columns as fractions of image width: centre third = straight ahead of the car
        self.col_left_frac = self.declare_parameter('col_left_frac', 0.33).value
        self.col_right_frac = self.declare_parameter('col_right_frac', 0.67).value

        # distance safety zone in front of the camera (replaces the LIDAR rectangle)
        # camera can't measure closer than ~0.3 m, so this must stay well above that
        self.front_padding = self.declare_parameter('front_padding', 0.60).value

        # TTC thresholds and speed multipliers for each braking stage
        # these are lower than the LIDAR version because the depth band only covers a short
        # distance ahead, so a high ttc_safe would brake during normal driving
        self.ttc_safe = self.declare_parameter('ttc_safe', 0.8).value
        self.ttc_hard_brake = self.declare_parameter('ttc_hard_brake', 0.6).value
        self.ttc_full_stop = self.declare_parameter('ttc_full_stop', 0.4).value
        self.mild_brake_factor = self.declare_parameter('mild_brake_factor', 0.60).value
        self.hard_brake_factor = self.declare_parameter('hard_brake_factor', 0.30).value

        # if no depth image arrives for this long, assume the camera died and stop
        self.image_timeout = self.declare_parameter('image_timeout', 0.3).value

        # hysteresis: after braking, the path must stay clear this long before driving again
        # (stops the stop/go/stop/go loop where stopping makes TTC look safe again instantly)
        self.release_time = self.declare_parameter('release_time', 0.5).value

        # initialize our speed and steering angle so that we have them from the start
        self.speed = 0.0
        self.steering_angle = 0.0
        self.last_image_time = self.get_clock().now().nanoseconds / 1e9
        self.last_safe = False  # latest safe/unsafe decision, starts unsafe until the first image
        self.last_unsafe_time = self.last_image_time  # when we last saw danger
        self.last_distance = np.inf  # latest distance ahead (only used by the commented-out brake logs)

        self.publisher_ = self.create_publisher(
                AckermannDriveStamped,
                '/safety_drive',
                10)
        # need to continuously publish a boolean on whether or not safety_node is operating or not
        self.status_publisher = self.create_publisher(
                Bool,
                '/safety_status',
                10)
        # sensor-data QoS works with the camera whether it publishes reliable or best-effort
        self.depth_subscription = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_reading,
                qos_profile_sensor_data)
        self.odom_subscription = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_reading,
                10)
        self.drive_subscription = self.create_subscription(
                AckermannDriveStamped,
                '/drive',
                self.drive_reading,
                10)

        # checks every 0.05 s that the camera is still sending images
        self.watchdog_timer = self.create_timer(0.05, self.camera_watchdog)


    def odom_reading(self, msg: Odometry) -> None:
        """
        Store the car's current forward speed.

        Args:
            msg (Odometry): Odometry message with the car's velocity.

        Returns:
            None

        Side effects:
            Updates self.speed with the negated x velocity in m/s (negative on the actual car, same as the LIDAR safety node)
        """
        # store the linear velocity in x into current speed
        self.speed = -msg.twist.twist.linear.x


    def drive_reading(self, msg: AckermannDriveStamped) -> None:
        """
        Store the current steering angle so braking keeps the same steering.

        Args:
            msg (AckermannDriveStamped): The active drive command.

        Returns:
            None

        Side effects:
            Updates self.steering_angle with the commanded angle in radians.
        """
        self.steering_angle = msg.drive.steering_angle


    def depth_to_array(self, msg: Image) -> np.ndarray:
        """
        Convert a 16-bit ROS depth image into a numpy array of distances.

        Args:
            msg (Image): Depth image with encoding 16UC1, each pixel a distance in millimetres.

        Returns:
            np.ndarray: uint16 array of shape (height, width), 0 means no reading.
        """
        dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
        flat = np.frombuffer(bytes(msg.data), dtype=dtype)
        # 'step' is bytes per row and can include padding, so trim to the real width
        return flat.reshape(msg.height, msg.step // 2)[:, :msg.width]


    def depth_reading(self, msg: Image) -> None:
        """
        Find the distance to the closest obstacle ahead and compute TTC from a depth image.

        Takes the 10th percentile of the measured pixels in the band (pixels reading 0 have
        no depth and are skipped). TTC is that distance divided by the current speed.

        Args:
            msg (Image): Depth image from the RealSense camera.

        Returns:
            None

        Side effects:
            Updates last_image_time and calls partial_braking(), which may publish to /safety_drive and always publishes to /safety_status
        """
        self.last_image_time = self.get_clock().now().nanoseconds / 1e9

        # a non-depth image means something is misconfigured, so treat it as unsafe
        if msg.encoding not in ('16UC1', 'mono16'):
            # self.get_logger().error(f'Expected 16UC1 depth image, got {msg.encoding}', throttle_duration_sec=2.0)
            self.partial_braking(0.0, True)
            return

        depth = self.depth_to_array(msg)
        width = depth.shape[1]

        # band of the image straight ahead of the car
        band = depth[self.row_top:self.row_bottom,
                     int(width * self.col_left_frac):int(width * self.col_right_frac)]

        # screening for valid depth pixels (0 = camera couldn't measure that pixel)
        valid = band[band > 0]

        if valid.size == 0:
            # nothing measurable ahead, treat as clear
            distance = np.inf
        else:
            # 10th percentile: reacts to the closest object without a few noisy pixels triggering it like min() would
            distance = np.percentile(valid, 10) / 1000.0  # mm -> m

        self.last_distance = distance

        # anything inside the padded zone in front of the car
        obstacle_in_box = bool(distance < self.front_padding)

        # calculate time to collision only when driving forward
        if self.speed > 0.0:
            min_ttc = distance / self.speed
        else:
            min_ttc = np.inf

        # pass min_ttc and obstacle_in_box onto partial-braking method
        self.partial_braking(min_ttc, obstacle_in_box)


    def camera_watchdog(self) -> None:
        """
        Stop the car if the camera stops sending depth images.

        Returns:
            None

        Side effects:
            If no image has arrived within image_timeout seconds, calls partial_braking() with a full stop; otherwise re-publishes the latest status to /safety_status
        """
        elapsed_time = self.get_clock().now().nanoseconds / 1e9 - self.last_image_time

        if elapsed_time > self.image_timeout:
            # self.get_logger().error('Depth camera went offline, applying emergency brake.', throttle_duration_sec=2.0)
            # TTC of 0 and obstacle_in_box True forces stage 3 (full stop)
            self.partial_braking(0.0, True)
        else:
            # camera is alive: re-send the latest status so master_control's 0.1 s timeout
            # never fires between camera frames (camera runs slower than the LIDAR did)
            self.status_publisher.publish(Bool(data=self.last_safe))


    def partial_braking(self, current_min_ttc: float, obstacle_in_box: bool) -> None:
        """
        Brake based on TTC and the safety zone, and publish safety status.

        Full stop if TTC < ttc_full_stop or an obstacle is in the safety zone, otherwise partial braking.
        After any braking, the car stays stopped until the path has been clear for release_time seconds.

        Args:
            current_min_ttc (float): Minimum time to collision in seconds.
            obstacle_in_box (bool): True if anything is closer than front_padding ahead of the car

        Returns:
            None

        Side effects:
            Publishes the safe/unsafe flag to /safety_status, and a brake command to /safety_drive when braking.
        """
        status_msg = Bool()
        now = self.get_clock().now().nanoseconds / 1e9
        danger = current_min_ttc < self.ttc_safe or obstacle_in_box

        if danger:
            self.last_unsafe_time = now

        # safe state: only once the path has stayed clear for release_time seconds
        if not danger and now - self.last_unsafe_time >= self.release_time:
            status_msg.data = True
            self.last_safe = True
            self.status_publisher.publish(status_msg)
            return

        # clear now, but still inside the hold time after braking: keep the car stopped
        if not danger:
            current_min_ttc = 0.0  # forces stage 3 below

        # every remaining case is unsafe
        status_msg.data = False
        self.last_safe = False
        self.status_publisher.publish(status_msg)

        # build the drive message since stages 1-3 all use it
        brake_msg = AckermannDriveStamped()
        brake_msg.header.stamp = self.get_clock().now().to_msg()
        brake_msg.header.frame_id = "camera"
        brake_msg.drive.steering_angle = self.steering_angle

        # stage 3: full braking
        # triggers if TTC is less than ttc_full_stop OR if an object is inside the safety zone
        # (or we are still holding after a brake, see release_time above)
        if current_min_ttc < self.ttc_full_stop or obstacle_in_box:
            stage = 3
            target_speed = 0.0
        # stage 2: hard braking
        elif current_min_ttc < self.ttc_hard_brake:
            stage = 2
            target_speed = self.speed * self.hard_brake_factor
        # stage 1: mild braking
        else:
            stage = 1
            target_speed = self.speed * self.mild_brake_factor

        # the car can't drive slower than min_drive_speed, so any partial-braking below that becomes a full stop
        if stage < 3 and target_speed < self.min_drive_speed:
            # self.get_logger().info(
            #     f'Stage {stage} target {target_speed:.2f} m/s below minimum '
            #     f'{self.min_drive_speed} m/s. Full Stop Applied. (dist {self.last_distance:.2f} m)',
            #     throttle_duration_sec=2.0)
            target_speed = 0.0
        # elif stage == 3:
        #     self.get_logger().info(f'Stage 3: Full Stop Applied. (dist {self.last_distance:.2f} m, '
        #                            f'speed {self.speed:.2f} m/s)', throttle_duration_sec=2.0)
        # else:
        #     self.get_logger().info(f'Stage {stage} braking being applied ({target_speed:.2f} m/s)',
        #                            throttle_duration_sec=2.0)

        brake_msg.drive.speed = target_speed
        self.publisher_.publish(brake_msg)


def main(args: list | None = None) -> None:
    """
    Start the vision safety node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit.
    """
    rclpy.init(args=args)
    safety_node = VisionSafetyNode()
    rclpy.spin(safety_node)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()