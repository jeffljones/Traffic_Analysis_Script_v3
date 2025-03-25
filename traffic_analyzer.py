# traffic_analyzer.py - Main application for traffic analysis

import os
import cv2
import time
import numpy as np
import argparse
from pathlib import Path
import threading
import queue

# Import our modules
from config import load_config
from calibration import Calibrator, CalibrationResult
from detection import VehicleDetector
from tracker import VehicleTracker
from visualization import TrafficVisualizer
from sort import SORT

class TrafficAnalyzer:
    """Main traffic analysis application"""
    
    def __init__(self, config_path=None, video_path=None, model_path=None, output_path=None, gui_mode=False):
        """Initialize the traffic analyzer with configuration"""
        # Load configuration
        self.config = load_config(config_path)
        
        # Override config with command line arguments if provided
        if video_path:
            self.config['video']['default_path'] = video_path
        if model_path:
            self.config['model']['path'] = model_path
        if output_path:
            self.config['output']['report_path'] = output_path
            
        # Set processing flags
        self.is_running = False
        self.is_paused = False
        self.gui_mode = gui_mode
        
        # Initialize components (these will be set up later)
        self.calibrator = None
        self.detector = None
        self.tracker = None
        self.visualizer = None
        
        # Set up threading components
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
        # Initialize processing statistics
        self.fps = 0
        self.frame_count = 0
        self.processing_time = 0
        
    def setup(self):
        """Set up components based on configuration"""
        print("[Setup] Initializing traffic analyzer components...")
        
        # Initialize detector
        self.detector = VehicleDetector(self.config)
        
        # Initialize calibrator (this will be used during processing)
        self.calibrator = Calibrator(self.config)
        
        # Initialize visualizer
        self.visualizer = TrafficVisualizer(self.config)
        
        print("[Setup] Components initialized successfully")
    
    def _process_video_thread(self):
        """Background thread for processing video frames"""
        video_path = self.config['video']['default_path']
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return
        
        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = self.config['video'].get('fps', 30)
            
        # Set up tracker with first frame for calibration
        ret, first_frame = cap.read()
        if not ret:
            print("[Error] Could not read first frame from video")
            return
            
        # Perform calibration
        calibration_result = self.calibrator.perform_manual_calibration(first_frame)
        
        # Initialize tracker with calibration result
        self.tracker = VehicleTracker(self.config, calibration_result)
        
        # Process frames
        self.frame_count = 0
        last_time = time.time()
        processing_times = []
        
        # Create video writer if saving output
        writer = None
        if self.config['output'].get('save_video', False):
            output_path = self.config['output'].get('output_video_path', 'output_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.video_fps, 
                                   (self.frame_width, self.frame_height))
        
        while self.is_running:
            # Check if paused
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[Processing] End of video")
                break
                
            # Start timer for this frame
            start_time = time.time()
            
            # Get timestamp (seconds from start of video)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time = timestamp_ms / 1000.0 if timestamp_ms > 0 else self.frame_count / self.video_fps
            
            # Detect vehicles
            detections = self.detector.detect_vehicles(frame)
            
            # Update tracker
            tracking_results = self.tracker.update(frame, detections, current_time)
            
            # Update visualizer statistics
            self.visualizer.update_statistics(tracking_results)
            
            # Create visualization
            result_frame = self.visualizer.create_visualization(
                frame, tracking_results, detections, self.detector, self.tracker
            )
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Calculate FPS (average of last 30 frames)
            if len(processing_times) > 30:
                processing_times.pop(0)
            self.fps = 1.0 / (sum(processing_times) / len(processing_times))
            
            # Add processing info to frame
            cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (self.frame_width - 150, self.frame_height - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to output video if enabled
            if writer:
                writer.write(result_frame)
                
            # Put results in queue for display thread
            try:
                if not self.result_queue.full():
                    self.result_queue.put((result_frame, tracking_results, detections), block=False)
            except queue.Full:
                # Skip a frame if queue is full
                pass
                
            self.frame_count += 1
            
            # Limit processing rate if running faster than video framerate
            elapsed = time.time() - start_time
            min_frame_time = 1.0 / self.video_fps
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            
        # Generate final report
        self.visualizer.generate_report(tracking_results, self.config['output']['report_path'])
        
        print("[Processing] Video processing complete")
    
    def process_video(self):
        """Process video and show results"""
        if not self.detector or not self.calibrator:
            print("[Error] Components not initialized. Call setup() first.")
            return
            
        self.is_running = True
        self.is_paused = False
        
        if self.gui_mode:
            # Start processing in a separate thread
            self.processing_thread = threading.Thread(target=self._process_video_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # In main thread, display results and handle user input
            while self.is_running:
                try:
                    # Get processed frame from queue
                    result_frame, tracking_results, detections = self.result_queue.get(timeout=1.0)
                    
                    # Display the frame
                    cv2.imshow('Traffic Analysis', result_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        self.is_running = False
                    elif key == ord('p'):
                        self.is_paused = not self.is_paused
                        print(f"Video {'paused' if self.is_paused else 'resumed'}")
                    elif key == ord('d'):
                        self.visualizer.toggle_detection_display()
                    elif key == ord('t'):
                        self.visualizer.toggle_tracks_display()
                    elif key == ord('i'):
                        self.visualizer.toggle_info_display()
                    elif key == ord('s'):
                        self.visualizer.toggle_speed_chart()
                    elif key == ord('c'):
                        self.visualizer.toggle_class_chart()
                        
                except queue.Empty:
                    # No new frames, check if processing thread is still alive
                    if not self.processing_thread.is_alive():
                        self.is_running = False
                        
            # Cleanup
            cv2.destroyAllWindows()
            
            # Wait for processing thread to finish
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=3.0)
        else:
            # Run directly in the main thread (no GUI)
            self._process_video_thread()
    
    def process_single_frame(self, frame, timestamp=None):
        """Process a single frame and return results"""
        if not self.detector or not self.tracker:
            print("[Error] Components not initialized.")
            return None
            
        if timestamp is None:
            timestamp = time.time()
            
        # Detect vehicles
        detections = self.detector.detect_vehicles(frame)
        
        # Update tracker
        tracking_results = self.tracker.update(frame, detections, timestamp)
        
        # Update visualizer statistics
        self.visualizer.update_statistics(tracking_results)
        
        # Create visualization
        result_frame = self.visualizer.create_visualization(
            frame, tracking_results, detections, self.detector, self.tracker
        )
        
        return {
            'frame': result_frame,
            'tracking_results': tracking_results,
            'detections': detections
        }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Traffic Analysis System')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--video', '-v', type=str, help='Path to video file')
    parser.add_argument('--model', '-m', type=str, help='Path to YOLO model file')
    parser.add_argument('--output', '-o', type=str, help='Path to output report file')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI (batch mode)')
    
    args = parser.parse_args()
    
    # Create and set up the analyzer
    analyzer = TrafficAnalyzer(
        config_path=args.config,
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        gui_mode=not args.no_gui
    )
    
    analyzer.setup()
    analyzer.process_video()


if __name__ == '__main__':
    main()
