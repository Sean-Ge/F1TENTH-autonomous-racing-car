"""
Wall Follow Node for an Autonomous Wall Following Algorithm.

This module implements a wall-following algorithm that drives
the car and maintains a set distance from the right wall.
"""
# Group B6

import rclpy
from rclpy.node import Node
import numpy as np
import math
from typing import Sequence

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class WallFollow(Node):
    """
    ROS 2 node that follows the right wall using PD steering control.

    Subscribes to /scan and publishes drive commands to /wall_follow_drive.
    """

    def __init__(self) -> None:
        """
        Set up the subscription, publisher, and control parameters.

        Returns:
            None

        Side effects:
            Reads params for the PD gains, wall geometry, and base speed, and sets up the publisher and subscription.
        """
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # publishers and subscribers
        self.subscription = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            10
        )
        self.publisher = self.create_publisher(
            AckermannDriveStamped, 
            '/wall_follow_drive',
            10
        )
        
        # values are read from params.yaml
        # pd control gains
        self.kp = self.declare_parameter('kp', 1.0).value
        self.kd = self.declare_parameter('kd', 0.01).value

        # wall following geometry
        self.desired_distance = self.declare_parameter('desired_distance', 1.15).value
        self.proj_trajectory = self.declare_parameter('proj_trajectory', 0.5).value
        self.theta_rad = self.declare_parameter('theta_rad', 1.0).value

        # base velocity (m/s)
        self.velocity = self.declare_parameter('velocity', 0.7).value

        # pd control state
        self.prev_error = 0.0
        self.prev_steering_angle = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9


    def get_range(self, range_data: Sequence[float], angle_min: float, angle_increment: float,
                  target_angle: float) -> float | None:
        """
        Get the LIDAR range at a given angle.

        Args:
            range_data (Sequence[float]): Range readings in meters.
            angle_min (float): Angle of the first beam in radians.
            angle_increment (float): Angle between beams in radians.
            target_angle (float): Beam angle in radians with positive left, negative right.

        Returns:
            float | None: The range in meters, or None if the index is out of bounds or the reading is inf/NaN.
        """

        # finds the index of the specific beam at given target angle
        index = int((target_angle - angle_min) / angle_increment)
        
        # quick check to see if our index is valid
        if index < 0 or index >= len(range_data):
            return None
            
        # quick check to see if our values are inf or NaN
        if math.isinf(range_data[index]) or math.isnan(range_data[index]):
            return None
            
        return range_data[index]


    def get_error(self, range_data: Sequence[float], angle_min: float, angle_increment: float) -> float | None:
        """
        Compute the distance error from the right wall.

        Uses two LIDAR beams to estimate the car's angle to the wall, then projects ahead by L to predict the future distance.

        Args:
            range_data (Sequence[float]): Range readings in meters.
            angle_min (float): Angle of the first beam in radians.
            angle_increment (float): Angle between beams in radians.

        Returns:
            float | None: Projected distance minus desired distance, in meters, or None if either beam has no valid reading.
        """
        # ray b from slides, pointing perpendicular right
        angle_b = -math.pi / 2.0
        b = self.get_range(range_data, angle_min, angle_increment, angle_b)

        # ray a from slides, pointing right with an offset of theta
        angle_a = angle_b + self.theta_rad
        a = self.get_range(range_data, angle_min, angle_increment, angle_a)

        # passing up the "None" flag if we can't find a wall to the right
        if a is None or b is None:
            return None

        # car's orientation relative to the wall
        alpha = math.atan2((a * math.cos(self.theta_rad) - b), (a * math.sin(self.theta_rad)))
        
        # shortest distance to the wall (from lidar sensor)
        AB = b * math.cos(alpha)
        
        # projected trajectory using L * sine
        CD = AB + self.proj_trajectory * math.sin(alpha)
        
        # get overall error
        error = CD - self.desired_distance
        
        return error


    def scan_callback(self, msg: LaserScan) -> None:
        """
        Run one step of PD steering control for each LIDAR scan.

        Args:
            msg (LaserScan): LIDAR scan with ranges and beam angles.

        Returns:
            None

        Side effects:
            Publishes a drive command to /wall_follow_drive and updates prev_error, prev_time, and prev_steering_angle.
        """
        # getting the current time to use for our dt calculation
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time

        # ignores a couple edge cases and returns to prevent the math from collapsing
        if dt <= 0:
            return

        # get e(t)
        error = self.get_error(msg.ranges, msg.angle_min, msg.angle_increment)

        # if we see the None flag, then we don't change anything with our current steering
        if error is None:
            current_steering_angle = self.prev_steering_angle
            self.prev_error = None

        else :
            # handling edge case for the first frame after prev_error = None
            if self.prev_error is None:
                derivative = 0.0

            else :
                # employ PD control
                derivative = (error - self.prev_error) / dt
            v0 = (self.kp * error) + (self.kd * derivative)

            current_steering_angle = -v0
            self.prev_error = error

        # note: the steering angle isn't clamped here, so a big error can ask for more than the
        # servo can physically do (around +-0.4 rad). The VESC servo limits should cap it on the car,
        # but keep it in mind if the gains get raised

        # prepare drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg() # labels our drive_msg with a timestamp
        drive_msg.header.frame_id = "laser" # our reference point driving velocity and steering angle is the lidar
        
        drive_msg.drive.steering_angle = float(current_steering_angle)
        drive_msg.drive.speed = float(self.velocity)
        
        self.publisher.publish(drive_msg)
        
        # update continuous variables
        self.prev_time = current_time
        self.prev_steering_angle = current_steering_angle


def main(args: list | None = None) -> None:
    """
    Start the wall follow node and run it until shutdown.

    Args:
        args (list, optional): Command-line arguments for rclpy and default to None

    Returns:
        None

    Side effects:
        Starts ROS 2, runs the node, and shuts it down on exit.
    """
    rclpy.init(args=args)
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()
    