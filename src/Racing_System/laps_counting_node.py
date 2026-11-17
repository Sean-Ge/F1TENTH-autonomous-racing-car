"""
Lap counter that uses the camera to recognise the start line.

When the node starts it averages the first few camera frames into a picture of the start.
Every frame after that is compared to it:
    1. make_small: shrink the frame to a small, blurred grey image
    2. similarity_to_start: score how much it looks like the start picture (1 = identical)
    3. update_laps: once the view has looked different for a while (we've left the start) and
       then looks like the start again, count a lap

After target_laps laps it publishes /race_complete = True, and master_control_node stops the car.
It also publishes /lap_counter_ready = True once the start picture is saved, which is what
master_control_node waits for before letting the car leave the start.
"""

# Group B6

import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Float32, Bool

# size of the small image every frame is shrunk to (pixels)
SMALL_WIDTH = 80
SMALL_HEIGHT = 60


class LapCounter(Node):
    """
    ROS 2 node that counts laps by comparing camera frames to a picture of the start.

    Subscribes to the colour image topic. Publishes the lap count to /lap_count, the similarity
    score to /lap_similarity, the finished flag to /race_complete, and whether the start picture
    has been saved to /lap_counter_ready.
    """

    def __init__(self) -> None:
        """
        Set up parameters, lap state, publishers, the image subscription and the status timer.

        Returns:
            None

        Side effects:
            Reads params, creates the publishers and subscription, and starts a 10 Hz timer
            that keeps publishing the lap count and flags.
        """
        super().__init__('lap_counter_node')

        # values come from params.yaml, these defaults only get used if a value is missing there
        image_topic = self.declare_parameter('image_topic', '/camera/camera/color/image_raw').value

        # how many laps to drive before asking master_control to stop the car.
        # The launch file sets this from its laps:= argument, so 3 only applies when run on its own
        self.target_laps = int(self.declare_parameter('target_laps', 3).value)

        # how many frames are averaged into the start picture
        self.reference_frames = int(self.declare_parameter('reference_frames', 5).value)

        # the car never crosses the line in exactly the same place twice, so the start picture is
        # slid up to this many pixels left and right (of SMALL_WIDTH) and the best match is used.
        # 0 compares the two pictures exactly as they are
        self.max_shift_px = int(self.declare_parameter('max_shift_px', 8).value)

        # similarity at or above this counts as "this looks like the start"
        self.match_threshold = self.declare_parameter('match_threshold', 0.82).value
        # similarity below this counts as "we've moved away from the start"
        self.leave_threshold = self.declare_parameter('leave_threshold', 0.70).value

        # this many low-similarity frames in a row means we've really left the start
        self.leave_frames = int(self.declare_parameter('leave_frames', 5).value)
        # a lap needs required_matches matching frames out of the last match_window,
        # so one lucky frame isn't enough
        self.match_window = int(self.declare_parameter('match_window', 5).value)
        self.required_matches = int(self.declare_parameter('required_matches', 3).value)

        # no lap can be shorter than this (s), so a similar-looking spot elsewhere on the track
        # (the opposite straight of our oval) can't count. Keep it above half a lap
        self.min_lap_time = self.declare_parameter('min_lap_time', 12.0).value

        # warn if the camera sends nothing for this long (s)
        self.image_timeout = self.declare_parameter('image_timeout', 2.0).value

        self.bridge = CvBridge()

        # frames collected for the start picture, then the picture itself once there are enough
        self.reference_stack = []
        self.reference = None

        self.laps = 0
        self.armed = False     # True once we've left the start and can count the next return
        self.finished = False  # True once target_laps is reached

        self.leave_count = 0
        self.last_lap_time = None
        # match / no match for the most recent frames
        self.match_history = deque(maxlen=self.match_window)

        # camera health, so each problem is logged once and not on every frame
        self.last_image_time = time.monotonic()
        self.camera_silent = False
        self.bad_frame_reported = False

        self.lap_pub = self.create_publisher(Int32, '/lap_count', 10)
        # published every frame so the score can be watched or recorded while tuning thresholds
        self.similarity_pub = self.create_publisher(Float32, '/lap_similarity', 10)
        # master_control_node stops the car once this is True
        self.finished_pub = self.create_publisher(Bool, '/race_complete', 10)
        # master_control_node holds the car at the start until this is True
        self.ready_pub = self.create_publisher(Bool, '/lap_counter_ready', 10)

        # sensor-data QoS works with the camera whether it publishes reliable or best-effort
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)

        # keep re-sending the count and flags even between laps
        self.status_timer = self.create_timer(0.1, self.publish_status)

        self.get_logger().info(
            f'Lap counter started. Target: {self.target_laps} laps. Waiting for {image_topic}')

    def image_callback(self, msg: Image) -> None:
        """
        Handle one camera frame: build the start picture first, then count laps against it.

        Args:
            msg (Image): Frame from the camera.

        Returns:
            None

        Side effects:
            May add to or finish the start picture, publishes to /lap_similarity, and updates
            the lap state through update_laps().
        """
        self.last_image_time = time.monotonic()

        # once all laps are done there's nothing left to do
        if self.finished:
            return

        small = self.make_small(msg)
        if small is None:
            # a frame we can't use can't count toward leaving or matching, so start those over
            self.match_history.clear()
            self.leave_count = 0
            return

        if self.reference is None:
            self.build_reference(small)
            return

        similarity = self.similarity_to_start(small)
        self.similarity_pub.publish(Float32(data=similarity))
        self.update_laps(similarity)

    def make_small(self, msg: Image) -> np.ndarray | None:
        """
        Shrink a camera frame to a small, slightly blurred grey image.

        A tiny image is fast to compare and doesn't care about small details or noise.

        Args:
            msg (Image): Frame from the camera.

        Returns:
            np.ndarray | None: Grey image, shape (SMALL_HEIGHT, SMALL_WIDTH), float32. None if
            the frame can't be converted or is one flat colour (lens covered).

        Side effects:
            Logs the first conversion failure (and only the first).
        """
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as error:
            if not self.bad_frame_reported:
                self.get_logger().error(f'Cannot read camera frames: {error}')
                self.bad_frame_reported = True
            return None

        small = cv2.resize(gray, (SMALL_WIDTH, SMALL_HEIGHT), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)

        # nothing to compare in a flat image
        if float(small.std()) < 1.0:
            return None

        return small

    def build_reference(self, small: np.ndarray) -> None:
        """
        Collect the first frames and average them into the picture of the start.

        Averaging a few frames evens out camera noise and the exposure still settling on the
        very first frame. The car is held at the start by master_control_node until this is done.

        Args:
            small (np.ndarray): Small grey image from make_small().

        Returns:
            None

        Side effects:
            Adds to reference_stack, and once it holds reference_frames frames sets reference,
            starts the lap timer and logs once.
        """
        self.reference_stack.append(small)
        if len(self.reference_stack) < self.reference_frames:
            return

        self.reference = np.mean(self.reference_stack, axis=0).astype(np.float32)
        self.reference_stack = []
        self.last_lap_time = time.monotonic()
        self.get_logger().info('Start picture saved. Counting laps.')

    def similarity_to_start(self, small: np.ndarray) -> float:
        """
        Score how much a frame looks like the start picture.

        The middle of the start picture (its left and right max_shift_px columns cut off) is
        slid across the frame, and at each position the two are compared by their brightness
        patterns with the average brightness removed, so a brighter or darker frame still
        matches. The best position wins, which lets a frame taken a little to one side of the
        start, or at a slightly different angle, still match.

        Args:
            small (np.ndarray): Small grey image from make_small().

        Returns:
            float: Similarity from -1 to 1 (1 = identical, 0 = unrelated).
        """
        shift = self.max_shift_px
        middle_of_start = self.reference[:, shift:SMALL_WIDTH - shift]

        # one score per sideways position of the start picture over the frame
        scores = cv2.matchTemplate(small, middle_of_start, cv2.TM_CCOEFF_NORMED)
        best = float(np.nan_to_num(scores).max())
        return float(np.clip(best, -1.0, 1.0))

    def update_laps(self, similarity: float) -> None:
        """
        Update the leave/return state with one similarity score, and count a lap if it completes one.

        Args:
            similarity (float): How much the current frame looks like the start.

        Returns:
            None

        Side effects:
            Updates the leave/match state, and on a lap updates laps (and finished once
            target_laps is reached), logs it and publishes the status.
        """
        now = time.monotonic()

        # step 1: make sure we've actually left the start before looking for the return
        if not self.armed:
            if similarity < self.leave_threshold:
                self.leave_count += 1
            else:
                self.leave_count = 0

            if self.leave_count >= self.leave_frames:
                self.armed = True
                self.leave_count = 0
                self.match_history.clear()
            return

        # step 2: ignore anything that would make the lap shorter than min_lap_time
        if now - self.last_lap_time < self.min_lap_time:
            self.match_history.clear()
            return

        # step 3: count a lap when enough of the most recent frames look like the start
        self.match_history.append(similarity >= self.match_threshold)
        if len(self.match_history) < self.match_window:
            return
        if sum(self.match_history) < self.required_matches:
            return

        lap_time = now - self.last_lap_time
        self.laps += 1
        self.last_lap_time = now

        # have to leave the start again before the next lap can count
        self.armed = False
        self.leave_count = 0
        self.match_history.clear()

        self.get_logger().info(
            f'Lap {self.laps}/{self.target_laps} completed in {lap_time:.2f} s')

        # last lap done: flag it so master_control stops the car
        if self.laps >= self.target_laps:
            self.finished = True
            self.get_logger().info('Target laps reached. Requesting vehicle stop.')

        # send the new count right away instead of waiting for the timer
        self.publish_status()

    def publish_status(self) -> None:
        """
        Publish the lap count and flags, and check the camera is still sending frames.

        Called by the 10 Hz timer and right after a lap is counted.

        Returns:
            None

        Side effects:
            Publishes to /lap_count, /race_complete and /lap_counter_ready, and logs once when
            the camera goes silent and once when it comes back.
        """
        # sent over and over so master_control still gets them even if it starts late
        self.lap_pub.publish(Int32(data=self.laps))
        self.finished_pub.publish(Bool(data=self.finished))
        self.ready_pub.publish(Bool(data=self.reference is not None))

        silent = time.monotonic() - self.last_image_time > self.image_timeout
        if silent and not self.camera_silent:
            self.get_logger().warn('No camera frames arriving: laps cannot be counted.')
        elif self.camera_silent and not silent:
            self.get_logger().info('Camera frames arriving again.')
        self.camera_silent = silent


def main(args: list | None = None) -> None:
    """
    Start the lap counter node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit (including Ctrl+C).
    """
    rclpy.init(args=args)
    node = LapCounter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
