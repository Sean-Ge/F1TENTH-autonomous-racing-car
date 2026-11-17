"""
Master control node: the only node that publishes to /drive.

It forwards the follower's commands to the car, unless there is a reason to stop. The reasons
are checked in one place (find_stop_reason), in this order:
    1. the safety node has gone quiet
    2. the safety node says the path is not safe
    3. the lap counter says the race is complete
    4. the lap counter has not saved its start picture yet (start gate)
    5. the follower has gone quiet

While any of them holds, the car is commanded to zero speed with the wheels kept where the
follower wants them, so it still steers toward open space as it slows.
"""

# Group B6

import signal
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float32

FOLLOWER_TOPICS = {  # follower picked with the mode:= launch argument -> topic we forward from
    'gap': '/gap_follow_drive',
    'wall': '/wall_follow_drive',
    'vision': '/gap_follow_drive',  # vision_follow_node publishes on the gap topic
    'straight': '/gap_follow_drive',  # straight_drive_node (testing) publishes on the gap topic
}


class MasterControlNode(Node):
    """
    ROS 2 node that decides whether the follower may drive the car.

    Subscribes to /safety_status, /safety_speed_limit, the active follower's topic, /race_complete and
    /lap_counter_ready. Publishes to /drive.
    """

    def __init__(self) -> None:
        """
        Read the parameters, set the initial state, and create the publisher, subscriptions and timer.

        Returns:
            None

        Side effects:
            Declares params, creates the /drive publisher and the subscriptions, and starts the
            watchdog timer. The car starts stopped until every check passes.
        """
        super().__init__('master_control_node')

        # values come from params.yaml, these defaults only get used if a value is missing there
        # how long safety_node may go silent before we assume it died (s)
        self.timeout_threshold = self.declare_parameter('timeout_threshold', 0.1).value
        # how long the follower may go silent before we assume it died (s)
        self.follower_timeout = self.declare_parameter('follower_timeout', 0.1).value
        # how often the watchdog re-checks everything (s)
        self.safety_check_period = self.declare_parameter('safety_check_period', 0.01).value
        # hold the car at the start until the lap counter has saved its start picture
        self.wait_for_lap_counter = self.declare_parameter('wait_for_lap_counter', True).value
        # which follower to forward: gap, wall, vision or straight
        follow_mode = self.declare_parameter('follow_mode', 'gap').value

        # latest word from safety_node, and when it arrived (None until the first message)
        self.is_safe = False
        # fastest speed safety_node currently allows (m/s); no cap until it sends one
        self.speed_limit = float('inf')
        self.last_status_time = None

        # when the follower last sent a command, and the steering it asked for
        self.last_follower_time = None
        self.follower_steering = 0.0

        # both of these stay True once set: a finished race stays finished, and a started one
        # is never held at the start gate again
        self.race_complete = False
        self.lap_counter_ready = False

        # why the car is currently stopped (None while driving), so each stop is only logged once
        self.stop_reason = 'starting up'

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.safety_status_subscriber = self.create_subscription(
            Bool, '/safety_status', self.safety_status_callback, 10)
        self.speed_limit_subscriber = self.create_subscription(
            Float32, '/safety_speed_limit', self.speed_limit_callback, 10)
        self.race_complete_subscriber = self.create_subscription(
            Bool, '/race_complete', self.race_complete_callback, 10)
        self.lap_counter_ready_subscriber = self.create_subscription(
            Bool, '/lap_counter_ready', self.lap_counter_ready_callback, 10)

        # subscribe only to the active follower's topic
        if follow_mode in FOLLOWER_TOPICS:
            follower_topic = FOLLOWER_TOPICS[follow_mode]
            self.follower_subscriber = self.create_subscription(
                AckermannDriveStamped, follower_topic, self.follower_callback, 10)
            self.get_logger().info(f'Follow mode "{follow_mode}": forwarding {follower_topic}')
        else:
            # a spelling mistake in the mode: forward nothing, so the car stays stopped
            self.follower_subscriber = None
            self.get_logger().error(
                f'Unknown follow_mode "{follow_mode}", expected one of {list(FOLLOWER_TOPICS)}')

        # the watchdog: catches nodes going quiet, which no callback can notice
        self.watchdog_timer = self.create_timer(self.safety_check_period, self.watchdog)

    def now_s(self) -> float:
        """
        Get the current time from the node clock.

        Returns:
            float: Time in seconds.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def safety_status_callback(self, msg: Bool) -> None:
        """
        Store the latest safety flag, and stop the car straight away if it says unsafe.

        Args:
            msg (Bool): True if the path is safe, False if the car must stop.

        Returns:
            None

        Side effects:
            Updates is_safe and last_status_time, and publishes a stop to /drive when unsafe
            (without waiting for the next watchdog tick).
        """
        self.is_safe = msg.data
        self.last_status_time = self.now_s()

        if not self.is_safe:
            self.stop_if_needed()

    def speed_limit_callback(self, msg: Float32) -> None:
        """
        Store the fastest speed the safety node currently allows.

        Args:
            msg (Float32): Speed cap in m/s (infinity means no cap).

        Returns:
            None

        Side effects:
            Updates speed_limit.
        """
        self.speed_limit = msg.data

    def race_complete_callback(self, msg: Bool) -> None:
        """
        Remember that the lap counter has finished the requested number of laps.

        Args:
            msg (Bool): True once all laps are done.

        Returns:
            None

        Side effects:
            Sets race_complete to True (it is never set back to False).
        """
        if msg.data:
            self.race_complete = True

    def lap_counter_ready_callback(self, msg: Bool) -> None:
        """
        Remember that the lap counter has saved its picture of the start line.

        Args:
            msg (Bool): True once the start picture is saved.

        Returns:
            None

        Side effects:
            Sets lap_counter_ready to True (it is never set back to False).
        """
        if msg.data:
            self.lap_counter_ready = True

    def follower_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Forward the follower's command to the car, unless there is a reason to stop.

        Args:
            msg (AckermannDriveStamped): Drive command from the active follower.

        Returns:
            None

        Side effects:
            Updates last_follower_time and follower_steering, and publishes either msg or a stop
            to /drive.
        """
        self.last_follower_time = self.now_s()
        self.follower_steering = msg.drive.steering_angle

        if not self.stop_if_needed():
            # the follower drives, but never faster than the safety node allows
            msg.drive.speed = min(msg.drive.speed, self.speed_limit)
            self.drive_publisher.publish(msg)

    def watchdog(self) -> None:
        """
        Re-check every stop reason on a timer, so a node that has gone quiet is still caught.

        Returns:
            None

        Side effects:
            Publishes a stop to /drive while there is a reason to stop.
        """
        self.stop_if_needed()

    def find_stop_reason(self) -> str | None:
        """
        Decide whether the car must be stopped right now, and why.

        Returns:
            str | None: A short reason if the car must stop, or None if the follower may drive.
        """
        now = self.now_s()

        if self.last_status_time is None or now - self.last_status_time > self.timeout_threshold:
            return 'safety node is not reporting'

        if not self.is_safe:
            return 'safety node says unsafe'

        if self.race_complete:
            return 'all laps complete'

        if self.wait_for_lap_counter and not self.lap_counter_ready:
            return 'waiting for the lap counter to save its start picture'

        if (self.last_follower_time is None
                or now - self.last_follower_time > self.follower_timeout):
            return 'follower is not sending commands'

        return None

    def stop_if_needed(self) -> bool:
        """
        Stop the car if there is a reason to, and log when that reason changes.

        Returns:
            bool: True if a stop was published, False if the follower may drive.

        Side effects:
            Publishes a stop to /drive while there is a reason to stop, updates stop_reason, and
            logs once when the car is stopped or released (never on every call).
        """
        reason = self.find_stop_reason()

        if reason != self.stop_reason:
            if reason is None:
                self.get_logger().info('Driving: follower in control')
            else:
                self.get_logger().info(f'Stopped: {reason}')
            self.stop_reason = reason

        if reason is None:
            return False

        self.publish_stop()
        return True

    def publish_stop(self) -> None:
        """
        Publish a zero-speed command, keeping the wheels where the follower wants them.

        Returns:
            None

        Side effects:
            Publishes a stop command to /drive.
        """
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = float(self.follower_steering)
        self.drive_publisher.publish(stop_msg)

    def shutdown_stop(self) -> None:
        """
        Publish a zero-speed command several times before the node exits.

        Returns:
            None

        Side effects:
            Cancels the watchdog timer and publishes stop commands to /drive for half a second.
        """
        self.watchdog_timer.cancel()

        for _ in range(10):
            self.publish_stop()
            time.sleep(0.05)

        self.get_logger().info('Shutting down: full stop published to /drive.')


def raise_keyboard_interrupt(signum, frame) -> None:
    """
    Turn a terminate signal into the same exception Ctrl+C raises.

    Args:
        signum: Signal number (unused).
        frame: Stack frame the signal arrived in (unused).

    Raises:
        KeyboardInterrupt: Always, so main() can stop the car before exiting.
    """
    raise KeyboardInterrupt


def main(args: list | None = None) -> None:
    """
    Start the master control node, and stop the car before shutting down.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None.

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, publishes a stop to /drive on Ctrl+C or a terminate signal,
        and shuts down on exit.
    """
    # turn off rclpy's own signal handling so we can still publish after Ctrl+C,
    # and treat a terminate signal (what ros2 launch sends if Ctrl+C is slow) the same way
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    master_control_node = MasterControlNode()
    try:
        rclpy.spin(master_control_node)
    except KeyboardInterrupt:
        master_control_node.shutdown_stop()
    finally:
        master_control_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
