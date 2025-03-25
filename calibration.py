# calibration.py - Handles camera calibration

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class CalibrationResult:
    meters_per_pixel: float
    calibration_points: Optional[List[Tuple[int, int]]] = None
    perspective_matrix: Optional[np.ndarray] = None

class Calibrator:
    def __init__(self, config):
        self.config = config['calibration']
        
    def calculate_drone_calibration(self) -> float:
        """
        Calculate meters per pixel based on drone camera parameters
        """
        sensor_width_mm = self.config['sensor_width_mm']
        focal_length_mm = self.config['focal_length_mm']
        image_width_px = self.config['image_width_px']
        altitude_m = self.config['altitude_m']
        tilt_angle_deg = self.config['tilt_angle_deg']
        
        tilt_angle_rad = math.radians(tilt_angle_deg)
        gsd = (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px)
        adjusted_gsd = gsd / math.cos(tilt_angle_rad)
        
        print(f"[Calibration] Drone-based calibration factor: {adjusted_gsd:.6f} m/px")
        return adjusted_gsd
    
    def perform_manual_calibration(self, frame, real_world_distance_m=None) -> CalibrationResult:
        """
        Allow user to click points to manually calibrate meters per pixel
        
        Args:
            frame: Input image frame
            real_world_distance_m: Optional override for the known distance
        """
        # Use provided distance or fall back to config
        if real_world_distance_m is None:
            real_world_distance_m = self.config['known_distance_m']
            
        print(f"[Calibration] Starting manual calibration... Click two points representing {real_world_distance_m} meters.")
        
        points = []
        image_display = frame.copy()
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
                if len(points) == 1:
                    cv2.putText(image_display, "Point 1", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif len(points) == 2:
                    cv2.putText(image_display, "Point 2", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.line(image_display, points[0], points[1], (0, 255, 0), 2)
                    distance_px = np.linalg.norm(np.array(points[1]) - np.array(points[0]))
                    cv2.putText(image_display, f"{distance_px:.1f} px = {real_world_distance_m} m", 
                              ((points[0][0] + points[1][0])//2, (points[0][1] + points[1][1])//2 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Calibration", image_display)
        
        # Show calibration distance in the window title
        cv2.imshow(f"Calibration - Select two points representing {real_world_distance_m}m", image_display)
        cv2.setMouseCallback(f"Calibration - Select two points representing {real_world_distance_m}m", click_event)
        
        # Wait for user to select points
        while len(points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return CalibrationResult(meters_per_pixel=self.calculate_drone_calibration())
        
        cv2.waitKey(1000)  # Give user time to see the final calibration
        cv2.destroyAllWindows()
        
        if len(points) == 2:
            pixel_distance = np.linalg.norm(np.array(points[1]) - np.array(points[0]))
            scale_factor = real_world_distance_m / pixel_distance
            print(f"[Calibration] Manual calibration: {scale_factor:.6f} m/px")
            return CalibrationResult(meters_per_pixel=scale_factor, calibration_points=points)
        else:
            print("[Calibration] Calibration failed. Using drone calibration instead.")
            return CalibrationResult(meters_per_pixel=self.calculate_drone_calibration())
            
    def setup_perspective_transform(self, frame, real_world_distance_m=None) -> CalibrationResult:
        """
        Setup perspective transformation with 4 points for better speed calculation
        
        Args:
            frame: Input image frame
            real_world_distance_m: Optional override for the known distance
        """
        print("[Calibration] Setting up perspective transformation. Click 4 points to form a rectangle.")
        
        points = []
        image_display = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw guidelines for ideal point placement
        cv2.line(image_display, (w//3, 0), (w//3, h), (128, 128, 128), 1)
        cv2.line(image_display, (2*w//3, 0), (2*w//3, h), (128, 128, 128), 1)
        cv2.line(image_display, (0, h//3), (w, h//3), (128, 128, 128), 1)
        cv2.line(image_display, (0, 2*h//3), (w, 2*h//3), (128, 128, 128), 1)
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(image_display, f"P{len(points)}", (x+5, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if len(points) >= 2:
                        for i in range(len(points)-1):
                            cv2.line(image_display, points[i], points[i+1], (0, 0, 255), 1)
                            
                    if len(points) == 4:
                        cv2.line(image_display, points[3], points[0], (0, 0, 255), 1)
                        
                    cv2.imshow("Perspective Calibration", image_display)
        
        cv2.imshow("Perspective Calibration", image_display)
        cv2.setMouseCallback("Perspective Calibration", click_event)
        
        # Instructions
        instructions = [
            "Click 4 points to define a rectangle in real-world coordinates.",
            "Points should be in clockwise order starting from top-left.",
            "Press ESC to cancel and use simple calibration instead."
        ]
        
        for i, line in enumerate(instructions):
            cv2.putText(image_display, line, (10, 30 + i * 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image_display, line, (10, 30 + i * 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.imshow("Perspective Calibration", image_display)
        
        # Wait for user to select 4 points
        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyWindow("Perspective Calibration")
                # Fall back to manual calibration
                return self.perform_manual_calibration(frame, real_world_distance_m)
        
        cv2.waitKey(1000)  # Give user time to see the final calibration
        cv2.destroyWindow("Perspective Calibration")
        
        # Define destination points (rectangle in top-down view)
        # Estimate the real-world rectangle dimensions based on selected points
        src_points = np.array(points, dtype=np.float32)
        
        # Calculate the width and height of the rectangle in the original image
        width1 = np.linalg.norm(src_points[1] - src_points[0])
        width2 = np.linalg.norm(src_points[2] - src_points[3])
        avg_width = (width1 + width2) / 2
        
        height1 = np.linalg.norm(src_points[3] - src_points[0])
        height2 = np.linalg.norm(src_points[2] - src_points[1])
        avg_height = (height1 + height2) / 2
        
        # Destination points (rectangle with same aspect ratio)
        dst_points = np.array([
            [0, 0],
            [avg_width, 0],
            [avg_width, avg_height],
            [0, avg_height]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # For calibration, use the average of manual calibration values
        calibration_result = self.perform_manual_calibration(frame, real_world_distance_m)
        calibration_result.perspective_matrix = perspective_matrix
        
        return calibration_result