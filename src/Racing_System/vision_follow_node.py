"""
Vision follow node: follow-the-gap using only the colour camera.

Every camera frame goes through the same steps:
    1. to_rgb: turn the ROS image into an RGB array
    2. floor_threshold: mark every pixel that doesn't look like floor (HSV colour check)
    3. clean_mask: remove specks, fill holes in obstacles, and ignore flat floor tape
    4. push_obstacles: grow obstacles sideways by half the car's width
    5. cast_rays: fan rays out from the bottom-centre pixel and see how far each gets
    6. choose_ray: steer toward the longest ray and pick a speed from its length.
       If no ray is usable, turn full left (our track runs counter-clockwise)
    7. publish_drive: send the command on /gap_follow_drive

The geometry (camera_model.py), mask (free_space.py) and ray (rays.py) code lives
in milestone3/vision/.
"""

# Group B6

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image

from milestone3.vision.camera_model import (
    COLOR_DEFAULT,
    CameraPose,
    first_in_range_row,
    push_px_per_row,
)
from milestone3.vision.free_space import (
    FreeSpaceConfig,
    clean_mask as fs_clean_mask,
    floor_threshold as fs_floor_threshold,
    push_obstacles as fs_push_obstacles,
)
from milestone3.vision.rays import (
    build_ray_table,
    cast_rays as rays_cast_rays,
)


class VisionFollow(Node):
    """
    ROS 2 node that finds free floor in the camera image and drives toward the longest open ray.

    Subscribes to /camera/camera/color/image_raw and publishes steering and speed commands to /gap_follow_drive.
    """

    def __init__(self) -> None:
        """
        Declare parameters, set up the subscription and publisher, and build the lookup tables.

        Returns:
            None

        Side effects:
            Declares params, sets blind_since_s to None, creates the image subscription and the
            /gap_follow_drive publisher, and calls build_tables().
        """
        super().__init__('vision_follow_node')

        # camera mount: measured from the depth images of run_1506 after the camera was re-mounted level.
        # every distance, the push size and the ray bearings depend on these, so re-measure if the camera moves
        self.declare_parameter('camera_height_m', 0.188)
        self.declare_parameter('camera_pitch_deg', -0.7)

        # how wide the car is, how far ahead we look, and how wide the ray fan is
        self.declare_parameter('half_width_m', 0.18)
        self.declare_parameter('max_range_m', 2.5)
        self.declare_parameter('ray_half_fov_deg', 85)

        # colour thresholds and clean-up kernel sizes for the floor mask
        self.declare_parameter('v_floor_min', 90)
        self.declare_parameter('s_floor_max', 40)
        self.declare_parameter('open_kernel', 3)
        self.declare_parameter('close_kernel', 7)
        self.declare_parameter('min_vertical_px', 6)

        # driving: steering limit, speed range, and what to do when no ray is usable
        self.declare_parameter('min_valid_reach_m', 0.35)
        self.declare_parameter('max_steering_rad', 0.40)
        self.declare_parameter('max_speed_mps', 1.5)
        self.declare_parameter('min_speed_mps', 0.5)
        self.declare_parameter('blind_speed_mps', 0.7)
        self.declare_parameter('blind_timeout_s', 3.0)
        self.declare_parameter('steering_offset_rad', 0.0)

        # time we first lost every ray (None while we can see a way forward)
        self.blind_since_s: Optional[float] = None

        # colour frames in, Ackermann drive commands out
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/gap_follow_drive', 10)

        # the camera geometry never changes while running, so do the heavy maths once
        self.build_tables()

    def build_tables(self) -> None:
        """
        Read the parameters and precompute everything that only depends on the camera geometry.

        That's the topmost row we use, how many pixels to grow obstacles by on each row,
        and the full ray lookup table.

        Returns:
            None

        Side effects:
            Sets the parameter attributes, intrinsics, pose, fs_cfg, top_row, push_px and ray_table.
        """
        # read everything here so this function has all it needs
        self.half_width_m = float(self.get_parameter('half_width_m').value)
        self.max_range_m = float(self.get_parameter('max_range_m').value)
        self.ray_half_fov_deg = int(self.get_parameter('ray_half_fov_deg').value)

        self.min_valid_reach_m = float(self.get_parameter('min_valid_reach_m').value)
        self.max_steering_rad = float(self.get_parameter('max_steering_rad').value)
        self.max_speed_mps = float(self.get_parameter('max_speed_mps').value)
        self.min_speed_mps = float(self.get_parameter('min_speed_mps').value)
        self.blind_speed_mps = float(self.get_parameter('blind_speed_mps').value)
        self.blind_timeout_s = float(self.get_parameter('blind_timeout_s').value)
        self.steering_offset_rad = float(self.get_parameter('steering_offset_rad').value)

        # standard D435 colour intrinsics, with the camera height and tilt from the parameters
        self.intrinsics = COLOR_DEFAULT
        self.pose = CameraPose(float(self.get_parameter('camera_height_m').value),
                               math.radians(float(self.get_parameter('camera_pitch_deg').value)))

        # floor mask settings from the parameters
        self.fs_cfg = FreeSpaceConfig(
            v_floor_min=int(self.get_parameter('v_floor_min').value),
            s_floor_max=int(self.get_parameter('s_floor_max').value),
            open_kernel=int(self.get_parameter('open_kernel').value),
            close_kernel=int(self.get_parameter('close_kernel').value),
            min_vertical_px=int(self.get_parameter('min_vertical_px').value),
        )

        # one ray per degree from -ray_half_fov_deg to +ray_half_fov_deg
        angles_deg = tuple(range(-self.ray_half_fov_deg, self.ray_half_fov_deg + 1))

        # highest image row that still sees floor within max_range_m; rows above it are ignored
        self.top_row = first_in_range_row(self.intrinsics, self.pose, self.max_range_m)

        # the car's half-width converted to pixels for every row
        self.push_px = push_px_per_row(self.intrinsics, self.pose, self.half_width_m)

        # pixel positions, floor distances and bearings along every ray
        self.ray_table = build_ray_table(
            self.intrinsics,
            self.pose,
            self.top_row,
            self.max_range_m,
            angles_deg=angles_deg,
        )

    def image_callback(self, msg: Image) -> None:
        """
        Run the whole vision pipeline on one camera frame and publish a drive command.

        Args:
            msg (Image): Colour frame from the camera.

        Returns:
            None

        Side effects:
            Publishes to /gap_follow_drive and may update blind_since_s.
        """
        # 1. ROS image -> (H, W, 3) RGB array
        rgb = self.to_rgb(msg)

        # 2. find what isn't floor, then clean up the mask
        nonfloor = self.floor_threshold(rgb)
        cleaned = self.clean_mask(nonfloor)

        # 3. grow obstacles sideways by half the car's width
        blocked = self.push_obstacles(cleaned)

        # 4. cast rays from the bottom-centre pixel
        reach_m, bearing_rad = self.cast_rays(blocked)

        # frame time from the image header, or the node's clock if the header has no stamp
        now_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if now_s <= 0.0:
            now_s = self.get_clock().now().nanoseconds * 1e-9

        # 5. pick a ray, work out steering and speed, and send it
        steering_rad, speed_mps = self.choose_ray(reach_m, bearing_rad, now_s)
        self.publish_drive(steering_rad, speed_mps)

        # debug summary of what the node saw and sent. The log call is commented out to keep the
        # terminal quiet; uncomment it to check whether a bad steering angle starts here or downstream
        best = float(np.max(reach_m))
        n_best = int(np.count_nonzero(np.isclose(reach_m, best, atol=1e-6)))
        # self.get_logger().info(
        #     f'img {rgb.shape[1]}x{rgb.shape[0]} (tables built for {self.intrinsics.width}x'
        #     f'{self.intrinsics.height}) | blocked strip {blocked[self.top_row:].mean():.0%} | '
        #     f'best reach {best:.2f} m on {n_best} ray(s) | steer {np.degrees(steering_rad):+.1f} deg '
        #     f'(sent with offset {np.degrees(steering_rad + self.steering_offset_rad):+.1f} deg) '
        #     f'speed {speed_mps:.2f} m/s', throttle_duration_sec=1.0)

    def to_rgb(self, msg: Image) -> np.ndarray:
        """
        Convert a ROS colour image into an RGB numpy array.

        Args:
            msg (Image): Image with encoding rgb8, bgr8, rgba8 or bgra8.

        Returns:
            np.ndarray: uint8 RGB array, shape (H, W, 3).

        Raises:
            ValueError: If the image uses any other encoding.
        """
        h, w = msg.height, msg.width
        enc = msg.encoding.lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        # rows can have padding on the end (msg.step is the real row length in bytes), so cut it off
        if enc in ('rgb8', 'bgr8'):
            arr = buf.reshape((h, msg.step))[:, : w * 3].reshape((h, w, 3))
            if enc == 'bgr8':
                return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr.copy()
        if enc in ('rgba8', 'bgra8'):
            arr = buf.reshape((h, msg.step))[:, : w * 4].reshape((h, w, 4))
            code = cv2.COLOR_BGRA2RGB if enc == 'bgra8' else cv2.COLOR_RGBA2RGB
            return cv2.cvtColor(arr, code)
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')

    def floor_threshold(self, rgb: np.ndarray) -> np.ndarray:
        """
        Mark every pixel that doesn't look like floor, using an HSV colour check.

        Args:
            rgb (np.ndarray): RGB image, shape (H, W, 3).

        Returns:
            np.ndarray: Boolean mask, True = not floor.
        """
        return fs_floor_threshold(rgb, self.fs_cfg)

    def clean_mask(self, nonfloor: np.ndarray) -> np.ndarray:
        """
        Clean up the non-floor mask: remove specks, fill holes, and drop flat floor tape.

        Args:
            nonfloor (np.ndarray): Mask from floor_threshold().

        Returns:
            np.ndarray: Cleaned boolean mask, True = not floor.
        """
        return fs_clean_mask(nonfloor, self.fs_cfg)

    def push_obstacles(self, nonfloor: np.ndarray) -> np.ndarray:
        """
        Grow obstacles sideways on each row by the car's half-width in pixels.

        Args:
            nonfloor (np.ndarray): Mask from clean_mask().

        Returns:
            np.ndarray: Boolean mask, True where the car's centre can't go.
        """
        return fs_push_obstacles(nonfloor, self.push_px, self.top_row)

    def cast_rays(self, blocked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cast every ray across the blocked mask.

        Args:
            blocked (np.ndarray): Mask from push_obstacles().

        Returns:
            Tuple[np.ndarray, np.ndarray]: (reach_m, bearing_rad) for each ray, bearing positive = left.
        """
        res = rays_cast_rays(self.ray_table, blocked)
        return res.reach_m, res.bearing_rad

    def choose_ray(self, reach_m: np.ndarray, bearing_rad: np.ndarray,
                   now_s: float) -> Tuple[float, float]:
        """
        Pick the longest ray and turn it into a steering angle and speed.

        If several rays tie for longest, the middle one wins. If no ray reaches
        min_valid_reach_m, the car turns full left at blind_speed_mps, and stops after
        blind_timeout_s of that.

        Args:
            reach_m (np.ndarray): Free distance of each ray in metres.
            bearing_rad (np.ndarray): Direction of each ray in radians, positive = left.
            now_s (float): Frame time in seconds.

        Returns:
            Tuple[float, float]: (steering_rad, speed_mps)

        Side effects:
            Sets or clears blind_since_s.
        """
        best_reach = float(np.max(reach_m))

        # no usable ray. This mostly happens mid-corner: the wall ahead (grown by the car's width)
        # covers the pixel every ray starts from, so they all die at once. We drive the track
        # counter-clockwise, so the way out is to the left: turn full left and keep creeping.
        # Only stop if it lasts a long time (lens covered, car stuck)
        if best_reach < self.min_valid_reach_m:
            if self.blind_since_s is None:
                self.blind_since_s = now_s
            if now_s - self.blind_since_s <= self.blind_timeout_s:
                return self.max_steering_rad, self.blind_speed_mps
            return self.max_steering_rad, 0.0
        self.blind_since_s = None

        # several rays can tie, e.g. every open ray gets exactly max_range_m.
        # taking the middle one keeps the car driving straight down an open corridor
        tied_indices = np.flatnonzero(np.isclose(reach_m, best_reach, atol=1e-6))
        best_idx = int(tied_indices[len(tied_indices) // 2])

        # steer straight at that ray, within the steering limit
        steering_rad = float(
            np.clip(bearing_rad[best_idx], -self.max_steering_rad, self.max_steering_rad)
        )

        # more free floor means more speed: scale linearly from min_speed_mps up to max_speed_mps
        span = max(self.max_range_m - self.min_valid_reach_m, 1e-6)
        reach_ratio = float(np.clip((best_reach - self.min_valid_reach_m) / span, 0.0, 1.0))
        speed_mps = float(
            self.min_speed_mps + (self.max_speed_mps - self.min_speed_mps) * reach_ratio
        )

        return steering_rad, speed_mps

    def publish_drive(self, steering_rad: float, speed_mps: float) -> None:
        """
        Publish a drive command on /gap_follow_drive.

        Args:
            steering_rad (float): Steering angle in radians, positive = left.
            speed_mps (float): Speed in m/s.

        Returns:
            None

        Side effects:
            Publishes an AckermannDriveStamped message to /gap_follow_drive.
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # servo trim: if the wheels pull to one side when told to go straight, steering_offset_rad
        # (positive = left) gets added to every command so 0 really means straight
        msg.drive.steering_angle = float(steering_rad + self.steering_offset_rad)
        msg.drive.speed = float(speed_mps)
        self.publisher.publish(msg)


def main(args: Optional[list] = None) -> None:
    """
    Start the vision follow node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit.
    """
    rclpy.init(args=args)
    vision_follow_node = VisionFollow()
    rclpy.spin(vision_follow_node)
    vision_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
