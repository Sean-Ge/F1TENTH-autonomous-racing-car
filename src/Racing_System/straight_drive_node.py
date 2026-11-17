"""
Straight drive node: a test follower that drives straight ahead at one fixed speed.

It does not look at the LIDAR at all. It exists so the safety node can be tested (for example
the brake test against a box) without the gap follower's steering and speed choices getting in
the way. It publishes on /gap_follow_drive, so master_control_node and safety_node treat it
exactly like the gap follower.
"""

# Group B6

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped

# how often the command is sent (s). Must stay well under master_control's follower_timeout
PUBLISH_PERIOD_S = 0.025


class StraightDrive(Node):
    """
    ROS 2 node that keeps commanding one speed with a fixed steering angle.

    Publishes to /gap_follow_drive.
    """

    def __init__(self) -> None:
        """
        Read the parameters, create the publisher, and start the timer that sends the command.

        Returns:
            None

        Side effects:
            Declares params, creates the /gap_follow_drive publisher and starts a 40 Hz timer.
        """
        super().__init__('straight_drive_node')

        # values come from params.yaml (speed can also be set with the speed:= launch argument)
        self.speed_mps = float(self.declare_parameter('speed_mps', 1.0).value)
        # servo trim: adjust until the car really drives straight (+ = left)
        self.steering_rad = float(self.declare_parameter('steering_rad', 0.0).value)

        self.publisher = self.create_publisher(AckermannDriveStamped, '/gap_follow_drive', 10)
        self.timer = self.create_timer(PUBLISH_PERIOD_S, self.publish_drive)

        self.get_logger().info(
            f'Straight drive: {self.speed_mps} m/s, steering {self.steering_rad} rad')

    def publish_drive(self) -> None:
        """
        Publish the fixed speed and steering angle.

        Returns:
            None

        Side effects:
            Publishes an AckermannDriveStamped message to /gap_follow_drive.
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.drive.speed = self.speed_mps
        msg.drive.steering_angle = self.steering_rad
        self.publisher.publish(msg)


def main(args: list | None = None) -> None:
    """
    Start the straight drive node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit (including Ctrl+C).
    """
    rclpy.init(args=args)
    straight_drive_node = StraightDrive()

    try:
        rclpy.spin(straight_drive_node)
    except KeyboardInterrupt:
        pass
    finally:
        straight_drive_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
