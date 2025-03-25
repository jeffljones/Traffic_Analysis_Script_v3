# tracker.py - Vehicle tracking with SORT

import numpy as np
import cv2
from sort import SORT
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import time

@dataclass
class TrackedVehicle:
    """Class to store information about a tracked vehicle"""
    id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    class_id: Optional[int] = None
    class_name: str = "vehicle"
    speed_mph: float = 0.0
    counted: bool = False
    last_update_time: float = 0.0


class VehicleTracker:
    """Track vehicles using SORT algorithm and calculate speeds"""
    
    def __init__(self, config, calibration_result):
        self.config = config
        self.meters_per_pixel = calibration_result.meters_per_pixel
        self.perspective_matrix = calibration_result.perspective_matrix
        
        # Initialize SORT tracker
        tracking_config = config['tracking']
        self.sort_tracker = SORT(
            max_age=int(tracking_config.get('max_age', 30)),
            min_hits=int(tracking_config.get('min_hits', 3)),
            iou_threshold=float(tracking_config.get('iou_threshold', 0.3))
        )
        
        # Initialize counters and storage
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.vehicle_speeds = []
        
        # Define the ROI line for counting (center of frame by default)
        self.roi_x = None  # Will be set when first frame is processed
        
        # Track when vehicles crossed the ROI
        self.counted_ids = set()
        
        # Track vehicle classes
        self.vehicle_classes = {}
        
    def set_roi(self, x_position):
        """Set the position of the ROI line"""
        self.roi_x = x_position
        
    def _convert_detections_to_sort_format(self, detections):
        """Convert our detection format to SORT format [x1, y1, x2, y2, score]"""
        if not detections:
            return np.empty((0, 5))
            
        sort_detections = []
        for i, detection in enumerate(detections):
            bbox = detection.bbox
            sort_detections.append([
                bbox[0], bbox[1], bbox[2], bbox[3], detection.confidence
            ])
            
        return np.array(sort_detections)
    
    def _calculate_speed(self, vehicle_id, current_time, current_bbox):
        """Calculate vehicle speed using Kalman filter velocity state"""
        tracker = None
        for t in self.sort_tracker.trackers:
            if t.id == vehicle_id:
                tracker = t
                break
                
        if tracker is None:
            return 0.0
            
        # Get velocity from Kalman filter
        velocity = tracker.get_velocity()
        speed_pixels_per_frame = np.linalg.norm(velocity)
        
        # Convert to real-world speed
        fps = self.config['video'].get('fps', 30)
        speed_mps = speed_pixels_per_frame * self.meters_per_pixel * fps
        speed_mph = speed_mps * 2.23694  # Convert m/s to mph
        
        # Apply exponential smoothing if we have previous speed
        if vehicle_id in self.tracked_vehicles:
            prev_speed = self.tracked_vehicles[vehicle_id].speed_mph
            alpha = self.config['tracking'].get('smoothing_alpha', 0.3)
            speed_mph = alpha * speed_mph + (1 - alpha) * prev_speed
            
        return speed_mph
        
    def _check_roi_crossing(self, vehicle_id, current_centroid):
        """Check if vehicle has crossed the ROI line"""
        if vehicle_id in self.counted_ids or self.roi_x is None:
            return False
            
        # Get previous centroid
        prev_centroid = None
        for t in self.sort_tracker.trackers:
            if t.id == vehicle_id and len(t.history) > 1:
                # Get previous predicted centroid from history
                prev_bbox = t.history[-2][0]
                prev_centroid = (
                    int((prev_bbox[0] + prev_bbox[2]) / 2),
                    int((prev_bbox[1] + prev_bbox[3]) / 2)
                )
                break
                
        if prev_centroid is None:
            return False
            
        # Check if crossed the ROI line (x-coordinate crosses the ROI line)
        if (prev_centroid[0] - self.roi_x) * (current_centroid[0] - self.roi_x) <= 0:
            self.counted_ids.add(vehicle_id)
            self.vehicle_count += 1
            return True
            
        return False
    
    def update(self, frame, detections, current_time):
        """
        Update tracker with new detections
        
        Args:
            frame: Current video frame
            detections: List of Detection objects
            current_time: Current time in seconds
            
        Returns:
            Dict containing tracking results
        """
        # Set ROI if not already set
        if self.roi_x is None:
            self.roi_x = frame.shape[1] // 2
            
        # Convert detections to SORT format
        sort_detections = self._convert_detections_to_sort_format(detections)
        
        # Update SORT tracker
        sort_results = self.sort_tracker.update(sort_detections)
        
        # Create/update tracked vehicles
        current_vehicles = {}
        
        for i, detection in enumerate(detections):
            # Store class information for later use
            for t in self.sort_tracker.trackers:
                # Match detection with tracker by checking bbox overlap
                d_bbox = detection.bbox
                t_bbox = t.bbox
                if (d_bbox[0] < t_bbox[2] and d_bbox[2] > t_bbox[0] and
                    d_bbox[1] < t_bbox[3] and d_bbox[3] > t_bbox[1]):
                    t.class_id = detection.class_id
                    t.class_name = detection.class_name
                    t.detection_index = id(detection)
                    break
        
        # Process SORT results
        for result in sort_results:
            bbox = result[:4].astype(int)
            track_id = int(result[4])
            
            # Calculate centroid
            centroid = (
                int((bbox[0] + bbox[2]) / 2),
                int((bbox[1] + bbox[3]) / 2)
            )
            
            # Get class information
            class_id = None
            class_name = "vehicle"
            
            for t in self.sort_tracker.trackers:
                if t.id + 1 == track_id:  # +1 because SORT returns ID+1
                    class_id = t.class_id
                    class_name = t.class_name if t.class_name else "vehicle"
                    break
            
            # Calculate speed
            speed_mph = self._calculate_speed(track_id-1, current_time, bbox)
            
            # Check if vehicle crossed ROI line
            crossed = self._check_roi_crossing(track_id-1, centroid)
            counted = crossed or (track_id-1 in self.counted_ids)
            
            # Create tracked vehicle object
            vehicle = TrackedVehicle(
                id=track_id,
                bbox=tuple(bbox),
                centroid=centroid,
                class_id=class_id,
                class_name=class_name,
                speed_mph=speed_mph,
                counted=counted,
                last_update_time=current_time
            )
            
            current_vehicles[track_id] = vehicle
            
            # Record speed data
            if counted and speed_mph > 0:
                self.vehicle_speeds.append({
                    'vehicle_id': track_id,
                    'class': class_name,
                    'speed_mph': speed_mph,
                    'timestamp': current_time,
                    'direction': 'right' if centroid[0] > self.roi_x else 'left'
                })
                
        self.tracked_vehicles = current_vehicles
        
        return {
            'tracked_vehicles': self.tracked_vehicles,
            'vehicle_count': self.vehicle_count,
            'roi_x': self.roi_x,
            'speeds': self.vehicle_speeds
        }
        
    def draw_tracks(self, frame, tracking_results):
        """
        Draw tracking visualization on frame
        
        Args:
            frame: Input frame
            tracking_results: Results from update method
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Define colors for different vehicle classes
        colors = {
            'car': (0, 255, 0),       # Green
            'motorcycle': (255, 0, 0), # Blue 
            'bus': (0, 165, 255),      # Orange
            'truck': (0, 0, 255),      # Red
            'boat': (255, 0, 255),     # Purple
            'vehicle': (255, 255, 0)   # Cyan (default)
        }
        
        # Draw ROI line
        roi_x = tracking_results['roi_x']
        cv2.line(annotated_frame, (roi_x, 0), (roi_x, frame.shape[0]), (0, 0, 255), 2)
        
        # Draw vehicle count
        vehicle_count = tracking_results['vehicle_count']
        cv2.putText(annotated_frame, f"Count: {vehicle_count}", (50, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw tracked vehicles
        for vehicle_id, vehicle in tracking_results['tracked_vehicles'].items():
            bbox = vehicle.bbox
            centroid = vehicle.centroid
            class_name = vehicle.class_name
            speed_mph = vehicle.speed_mph
            counted = vehicle.counted
            
            # Get appropriate color
            color = colors.get(class_name.lower(), colors['vehicle'])
            
            # Draw bounding box (thicker for counted vehicles)
            thickness = 3 if counted else 2
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw centroid
            cv2.circle(annotated_frame, centroid, 4, (0, 255, 255), -1)
            
            # Draw ID and speed
            label = f"ID {vehicle_id}: {class_name}"
            if speed_mph > 0:
                label += f" {speed_mph:.1f} MPH"
                
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                      
            # Draw trajectory (if we have history)
            for t in self.sort_tracker.trackers:
                if t.id + 1 == vehicle_id and len(t.history) > 1:
                    # Draw last 10 positions
                    points = []
                    for i in range(min(10, len(t.history))):
                        trk_bbox = t.history[-i-1][0]
                        cx = int((trk_bbox[0] + trk_bbox[2]) / 2)
                        cy = int((trk_bbox[1] + trk_bbox[3]) / 2)
                        points.append((cx, cy))
                    
                    # Draw trajectory line
                    if len(points) >= 2:
                        for i in range(len(points) - 1):
                            cv2.line(annotated_frame, points[i], points[i+1], color, 1)
        
        return annotated_frame
