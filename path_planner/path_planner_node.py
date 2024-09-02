#!/usr/bin/env python3

import math
import rclpy
import numpy as np

from rclpy.node import Node
from lfs_msgs.msg import Map
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

class PathPlannerNode(Node):
    
    def __init__(self):
        super().__init__('path_planner_node')

        # Create a publisher for the Path message
        self.path_publisher_ = self.create_publisher(Path, "/path/global", 10)
        # Create a publisher for the MarkerArray
        self.marker_publisher_ = self.create_publisher(MarkerArray, 'path_markers', 10)
        # Create a subscriber for the Map message
        self.map_subscriber_ = self.create_subscription(Map, "/mapping/track_points", self.map_callback, 10)

        # Store previously published marker IDs (used for deletion)
        self.previous_marker_ids = []

        self.get_logger().info("Path planner node has started")

    # Signed area of a ploygon 
    def compute_signed_area(self, points: list[tuple]):
        """
        Computes the signed area of a polygon made up of points.

        Parameters
        ----------
        points : list[tuple]
            The points of the polygon where each tuple contains the x and y coordinates of each point.

        Returns
        -------
        int
            The signed area.
        """
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i][0], points[i][1]
            x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
            area += (x1 * y2 - y1 * x2)
        return area / 2.0
    
    # Generate a smooth path using cubic spline interpolation
    def path_smoother(self, path_points: list[tuple], sample_factor=5):
        """
        Generates a smoothened path by using cubic splines.

        Parameters
        ----------
        points : List[tuple]
            The points of the polygon where each tuple contains the x and y coordinates of each point.
        sample_factor : int
            How many points the smoothened path should have for each point in the original path

        Returns
        -------
        list[tuple]
            Another list of tuples with more dense points following a cubic spline.
        """
        path = np.array(path_points)
        cs_x = CubicSpline(range(len(path)), path[:, 0], bc_type='periodic')
        cs_y = CubicSpline(range(len(path)), path[:, 1], bc_type='periodic')

        # Sample the spline at a higher resolution
        num_samples = len(path)*sample_factor  # Number of samples along the path
        t_values = np.linspace(0, len(path) - 1, num_samples)
        smooth_x_points = cs_x(t_values)
        smooth_y_points = cs_y(t_values)
        return list(zip(smooth_x_points, smooth_y_points))
    
    def compute_optimal_path(self, left_x: list[float], left_y: list[float], right_x: list[float], right_y: list[float]): 
        """
        Using magic, this function returns the shortest possible path along a track.

        Parameters
        ----------
        left_x : List[float]
            The x-coordinates of all track markers on the left side of the track.
        left_y : List[float]
            The y-coordinates of all track markers on the left side of the track.
        right_x : List[float]
            The x-coordinates of all track markers on the right side of the track.
        right_y : List[float]
            The y-coordinates of all track markers on the right side of the track.

        Returns
        -------
        list[tuple]
            The points of the shortest path on the track
        """
        delta_x = left_x - right_x
        delta_y = left_y - right_y
    
        # Number of segments
        n = len(delta_x)

        # https://math.stackexchange.com/questions/444289/shortest-path-and-minimum-curvature-path-implementation
        # Compute HS and BS matricies
        HS = np.zeros((n,n))
        BS = np.zeros(n)
        for i in range(n-1):
            delta_x_0 = right_x[i+1]-right_x[i]
            delta_y_0 = right_y[i+1]-right_y[i]

            beta_i = np.array([delta_x[i+1], -delta_x[i]])
            gamma_i = np.array([delta_y[i+1], -delta_y[i]])

            HS_i = np.outer(beta_i, beta_i) + np.outer(gamma_i, gamma_i)
            BS_i = 2*delta_x_0*beta_i + 2*delta_y_0*gamma_i

            E = np.zeros((2,n))
            E[0, i+1] = 1
            E[1, i] = 1

            HS += np.dot(np.dot(E.T, HS_i), E)
            BS += np.dot(BS_i, E)

        # Objective function
        def objective(alpha):
            return np.dot(np.dot(alpha.T, HS), alpha) + np.dot(BS, alpha)


        # Initial guess for optimal path (straight line through midpoints)
        alpha0 = np.ones((n))*0.5

        # Set bounds to ensure path is inside of the track (plus some buffer)
        buffer = 0.05
        bounds = [(0+buffer, 1-buffer)] * len(alpha0)

        # Solve the optimization problem
        result = minimize(objective, alpha0, bounds=bounds)
        alpha_star = result.x

        # Extract the optimized path
        opt_path_x = right_x + alpha_star*(left_x-right_x)
        opt_path_y = right_y + alpha_star*(left_y - right_y)
        return list(zip(opt_path_x, opt_path_y))

    def map_callback(self, map: Map):

        # Create the Path message
        path_msg = Path()

        # Fill the header
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map' 

        # Find indices of orange, yellow and blue cones
        orange_indices = [i for i, color in enumerate(map.color) if color == 2]
        yellow_indices = [i for i, color in enumerate(map.color) if color == 1]
        blue_indices = [i for i, color in enumerate(map.color) if color == 0]

        # Extract positions of orange cones to find the start point
        orange_x = np.array(map.x)[orange_indices]
        orange_y = np.array(map.y)[orange_indices]
        start_x = orange_x.mean()
        start_y = orange_y.mean()

        # Extract yellow and blue cone positions
        yellow_x = np.array(map.x)[yellow_indices]
        yellow_y = np.array(map.y)[yellow_indices]
        blue_x = np.array(map.x)[blue_indices]
        blue_y = np.array(map.y)[blue_indices]

        # Initialize path points
        path_points = [(start_x, start_y)]

        # Determine the midpoints 
        #mid_x = (blue_x + yellow_x) / 2
        #mid_y = (blue_y + yellow_y) / 2
        #mid = list(zip(mid_x, mid_y))

        # Compute and append optimal path
        opt_path = self.compute_optimal_path(blue_x, blue_y, yellow_x, yellow_y)
        path_points.extend(opt_path)

        # Close the loop to form a lap
        path_points.append((start_x, start_y))

        # Determine clockwise direction
        area = self.compute_signed_area(path_points)
        clockwise = area < 0
        if not clockwise:
            path_points = np.flip(path_points, axis=0) # axis=0 ensures x and y points do not swap

        # Sample the path at higher resolution using cubic splines
        smooth_points = self.path_smoother(path_points)

        # Initialize PoseStamped messages and markers arrays
        poses = []
        markers = []

        # Delete previous markers
        for marker_id in self.previous_marker_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = 'path_arrows'
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETE
            markers.append(delete_marker)

        self.previous_marker_ids.clear()

        # Create PoseStamped message and markers
        for i, (x, y) in enumerate(smooth_points):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
        
            # Calculate orientation to follow the path
            if i < len(smooth_points) - 1:
                next_x, next_y = smooth_points[i + 1][0], smooth_points[i+1][1]
                dx = next_x - x
                dy = next_y - y
                angle = math.atan2(dy, dx)
                pose.pose.orientation.z = math.sin(angle / 2)
                pose.pose.orientation.w = math.cos(angle / 2)

            # Last orientation is aligned with the first
            else:
                pose.pose.orientation = poses[0].pose.orientation

            poses.append(pose)

            # Create an arrow marker for each pose
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'path_arrows'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = 0.2  # Arrow length
            marker.scale.y = 0.05  # Arrow width
            marker.scale.z = 0.05  # Arrow height
            marker.color.a = 1.0  # Alpha
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0  # Green
            marker.color.b = 0.0  # Blue

            self.previous_marker_ids.append(i)
            markers.append(marker)

        # Add the PoseStamped messages to the Path message
        path_msg.poses = poses

        # Create the MarkerArray message
        marker_array_msg = MarkerArray()
        marker_array_msg.markers = markers

        # Publish the Path and MarkerArray message
        self.path_publisher_.publish(path_msg)
        self.marker_publisher_.publish(marker_array_msg)
        self.get_logger().info('Published path with {} poses'.format(len(poses)))
        
        
def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down node gracefully...")
    finally:
        # Perform any cleanup here
        if rclpy.ok():  # Check if shutdown has not already been called
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()