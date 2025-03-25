# detection.py - Vehicle detection using YOLO

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Detection:
    """Class to store detection information"""
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Handle vehicle detection using YOLO"""
    
    def __init__(self, config):
        self.config = config
        self.model_path = config['model']['path']
        self.conf_threshold = config['model']['confidence']
        
        # Define class names for COCO dataset (subset related to vehicles)
        self.vehicle_classes = {
        0: 'Bus',
        1: 'Car',
        2: 'Medium Truck',
        3: 'Motorcycle',
        4: 'Pickup',
        5: 'Pickup-Trailer',
        6: 'Semi',
        7: 'Semi-Trailer',
        8: 'Van'

        }
                
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Detection] Using device: {self.device}")
        
        # Load model
        print(f"[Detection] Loading YOLO model from {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model.to(self.device)
            self.model_loaded = True
        except Exception as e:
            print(f"[Error] Failed to load YOLO model: {e}")
            self.model_loaded = False
    
    def detect_vehicles(self, frame, include_optional_classes=False):
        """
        Detect vehicles in the frame
        
        Args:
            frame: Input image
            include_optional_classes: Whether to include optional vehicle classes
            
        Returns:
            List of Detection objects
        """
        if not self.model_loaded:
            return []
            
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Define detection boundaries (skip leftmost and rightmost X% of the image)
        boundary_percent = self.config['tracking']['boundary_percent']
        left_boundary = int(frame_width * boundary_percent / 100)
        right_boundary = int(frame_width * (1 - boundary_percent / 100))
        
        # Run detection
        results = self.model(frame, conf=self.conf_threshold)
        
        detections = []
        
        # Process detections
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6]
                cls = int(cls)
                
                # Check if this is a vehicle class we're interested in
                valid_class = False
                class_name = "unknown"
                
                if cls in self.vehicle_classes:
                    valid_class = True
                    class_name = self.vehicle_classes[cls]
                elif include_optional_classes and cls in self.optional_classes:
                    valid_class = True
                    class_name = self.optional_classes[cls]
                    
                if valid_class:
                    # Calculate centroid
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Skip detections outside boundaries
                    if cx < left_boundary or cx > right_boundary:
                        continue
                        
                    # Create detection object
                    detection = Detection(
                        centroid=(cx, cy),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        class_id=cls,
                        class_name=class_name
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections, tracked_vehicles=None):
        """
        Draw detection bounding boxes and info on frame
        
        Args:
            frame: Input frame to draw on
            detections: List of Detection objects
            tracked_vehicles: Optional dictionary of tracked vehicles with IDs
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Define colors for different classes
        colors = {
            'car': (0, 255, 0),       # Green
            'motorcycle': (255, 0, 0), # Blue 
            'bus': (0, 165, 255),      # Orange
            'truck': (0, 0, 255),      # Red
            'boat': (255, 0, 255)      # Purple
        }
        
        default_color = (255, 255, 0)  # Cyan for other classes
        
        # Draw each detection
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            class_name = detection.class_name
            color = colors.get(class_name, default_color)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # If we have tracking info, use it
            if tracked_vehicles is not None:
                # Try to find a tracked vehicle that matches this detection
                for vehicle_id, vehicle_data in tracked_vehicles.items():
                    if 'detection_index' in vehicle_data and vehicle_data['detection_index'] == id(detection):
                        label = f"ID {vehicle_id}: {class_name}"
                        if 'speed' in vehicle_data:
                            label += f" {vehicle_data['speed']:.1f} MPH"
                        
                        # Draw label
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        break
            else:
                # Just show class and confidence
                cv2.putText(annotated_frame, f"{class_name} {detection.confidence:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
