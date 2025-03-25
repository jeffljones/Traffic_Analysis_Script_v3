# traffic_analyzer_gui.py - GUI application for traffic analysis

import sys
import os
import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox, 
                           QTabWidget, QGroupBox, QSlider, QSpinBox, QDoubleSpinBox, 
                           QProgressBar, QMessageBox, QSplitter, QTextEdit, QStatusBar,
                           QAction, QMenu, QToolBar)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QThread

# Import our modules
from config import load_config
from calibration import Calibrator, CalibrationResult
from detection import VehicleDetector
from tracker import VehicleTracker
from visualization import TrafficVisualizer
from traffic_analyzer import TrafficAnalyzer

class VideoThread(QThread):
    """Thread for processing video frames"""
    update_frame = pyqtSignal(np.ndarray, dict, list)
    processing_finished = pyqtSignal()
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.is_running = False
        self.is_paused = False
        
    def run(self):
        """Process video frames"""
        try:
            video_path = self.analyzer.config['video']['default_path']
            self.status_update.emit(f"Opening video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {video_path}")
                return
                
            # Get video properties
            self.analyzer.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.analyzer.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.analyzer.video_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.analyzer.video_fps <= 0:
                self.analyzer.video_fps = self.analyzer.config['video'].get('fps', 30)
                
            # Get total frames for progress tracking
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 1000  # Fallback if can't determine frame count
                
            # Set up tracker with first frame for calibration
            ret, first_frame = cap.read()
            if not ret:
                self.error_occurred.emit("Could not read first frame from video")
                return
                
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            # Process frames
            self.analyzer.frame_count = 0
            last_time = time.time()
            processing_times = []
            
            self.is_running = True
            
            while self.is_running:
                # Check if paused
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                    
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    self.status_update.emit("End of video")
                    break
                    
                # Start timer for this frame
                start_time = time.time()
                
                # Get timestamp (seconds from start of video)
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time = timestamp_ms / 1000.0 if timestamp_ms > 0 else self.analyzer.frame_count / self.analyzer.video_fps
                
                # Detect vehicles
                detections = self.analyzer.detector.detect_vehicles(frame)
                
                # Update tracker
                tracking_results = self.analyzer.tracker.update(frame, detections, current_time)
                
                # Update visualizer statistics
                self.analyzer.visualizer.update_statistics(tracking_results)
                
                # Create visualization
                result_frame = self.analyzer.visualizer.create_visualization(
                    frame, tracking_results, detections, self.analyzer.detector, self.analyzer.tracker
                )
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                # Calculate FPS (average of last 30 frames)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                self.analyzer.fps = 1.0 / (sum(processing_times) / len(processing_times))
                
                # Update progress
                progress = int(100 * self.analyzer.frame_count / total_frames)
                self.progress_update.emit(progress)
                
                # Update status with FPS
                self.status_update.emit(f"Processing: {self.analyzer.fps:.1f} FPS | Frame {self.analyzer.frame_count}/{total_frames}")
                
                # Send frame to GUI
                self.update_frame.emit(result_frame, tracking_results, detections)
                
                self.analyzer.frame_count += 1
                
                # Limit processing rate to match video framerate or slightly slower
                elapsed = time.time() - start_time
                min_frame_time = 1.0 / self.analyzer.video_fps
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
            
            # Cleanup
            cap.release()
            
            # Generate final report
            self.analyzer.visualizer.generate_report(tracking_results, self.analyzer.config['output']['report_path'])
            self.status_update.emit(f"Report saved to {self.analyzer.config['output']['report_path']}")
            
            self.processing_finished.emit()
        
        except Exception as e:
            self.error_occurred.emit(f"Error during processing: {str(e)}")
        
    def stop(self):
        """Stop the thread"""
        self.is_running = False
        self.wait()
        
    def pause(self):
        """Pause processing"""
        self.is_paused = True
        
    def resume(self):
        """Resume processing"""
        self.is_paused = False


class CalibrationDialog(QWidget):
    """Dialog for calibration setup"""
    calibration_complete = pyqtSignal(CalibrationResult)
    
    def __init__(self, analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.calibrator = analyzer.calibrator
        self.calibration_result = None
        self.calibration_distance = None
        
        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Camera Calibration Setup")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(instructions)
        
        # Calibration options
        options_group = QGroupBox("Calibration Options")
        options_layout = QVBoxLayout()
        
        # Drone calibration parameters
        drone_group = QGroupBox("Drone Parameters")
        drone_layout = QHBoxLayout()
        
        # Sensor width
        sensor_width_layout = QVBoxLayout()
        sensor_width_label = QLabel("Sensor Width (mm):")
        self.sensor_width_input = QDoubleSpinBox()
        self.sensor_width_input.setRange(1.0, 50.0)
        self.sensor_width_input.setValue(self.analyzer.config['calibration']['sensor_width_mm'])
        sensor_width_layout.addWidget(sensor_width_label)
        sensor_width_layout.addWidget(self.sensor_width_input)
        drone_layout.addLayout(sensor_width_layout)
        
        # Focal length
        focal_length_layout = QVBoxLayout()
        focal_length_label = QLabel("Focal Length (mm):")
        self.focal_length_input = QDoubleSpinBox()
        self.focal_length_input.setRange(1.0, 300.0)
        self.focal_length_input.setValue(self.analyzer.config['calibration']['focal_length_mm'])
        focal_length_layout.addWidget(focal_length_label)
        focal_length_layout.addWidget(self.focal_length_input)
        drone_layout.addLayout(focal_length_layout)
        
        # Altitude
        altitude_layout = QVBoxLayout()
        altitude_label = QLabel("Altitude (m):")
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(1.0, 500.0)
        self.altitude_input.setValue(self.analyzer.config['calibration']['altitude_m'])
        altitude_layout.addWidget(altitude_label)
        altitude_layout.addWidget(self.altitude_input)
        drone_layout.addLayout(altitude_layout)
        
        # Tilt angle
        tilt_angle_layout = QVBoxLayout()
        tilt_angle_label = QLabel("Tilt Angle (deg):")
        self.tilt_angle_input = QDoubleSpinBox()
        self.tilt_angle_input.setRange(0.0, 90.0)
        self.tilt_angle_input.setValue(self.analyzer.config['calibration']['tilt_angle_deg'])
        tilt_angle_layout.addWidget(tilt_angle_label)
        tilt_angle_layout.addWidget(self.tilt_angle_input)
        drone_layout.addLayout(tilt_angle_layout)
        
        drone_group.setLayout(drone_layout)
        options_layout.addWidget(drone_group)
        
        # Manual calibration parameters
        manual_group = QGroupBox("Manual Calibration")
        manual_layout = QHBoxLayout()
        
        # Known distance
        known_distance_layout = QVBoxLayout()
        known_distance_label = QLabel("Known Distance (m):")
        self.known_distance_input = QDoubleSpinBox()
        self.known_distance_input.setRange(1.0, 100.0)
        self.known_distance_input.setValue(self.analyzer.config['calibration']['known_distance_m'])
        known_distance_layout.addWidget(known_distance_label)
        known_distance_layout.addWidget(self.known_distance_input)
        manual_layout.addLayout(known_distance_layout)
        
        manual_group.setLayout(manual_layout)
        options_layout.addWidget(manual_group)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Calibration buttons
        buttons_layout = QHBoxLayout()
        
        self.drone_calibration_btn = QPushButton("Use Drone Calibration")
        self.drone_calibration_btn.clicked.connect(self.use_drone_calibration)
        buttons_layout.addWidget(self.drone_calibration_btn)
        
        self.manual_calibration_btn = QPushButton("Perform Manual Calibration")
        self.manual_calibration_btn.clicked.connect(self.start_manual_calibration)
        buttons_layout.addWidget(self.manual_calibration_btn)
        
        self.perspective_calibration_btn = QPushButton("Perspective Calibration")
        self.perspective_calibration_btn.clicked.connect(self.start_perspective_calibration)
        buttons_layout.addWidget(self.perspective_calibration_btn)
        
        layout.addLayout(buttons_layout)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        # Status and result
        self.status_label = QLabel("Ready for calibration")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Preview frame
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 480)
        layout.addWidget(self.frame_label)
        
        self.setLayout(layout)
        
        # Load first frame
        self.load_preview_frame()
    
    def load_preview_frame(self):
        """Load first frame from video for calibration"""
        try:
            video_path = self.analyzer.config['video']['default_path']
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.status_label.setText(f"Error: Cannot open video {video_path}")
                return
                
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.status_label.setText("Error: Could not read frame from video")
                return
                
            self.calibration_frame = frame
            self.display_frame(frame)
            
        except Exception as e:
            self.status_label.setText(f"Error loading preview: {str(e)}")
    
    def display_frame(self, frame):
        """Display frame in the label"""
        if frame is None:
            return
            
        # Resize frame to fit in label while maintaining aspect ratio
        h, w = frame.shape[:2]
        label_w, label_h = self.frame_label.width(), self.frame_label.height()
        
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Convert frame to QImage
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], 
                     frame_rgb.strides[0], QImage.Format_RGB888)
        
        self.frame_label.setPixmap(QPixmap.fromImage(image))
    
    def use_drone_calibration(self):
        """Use drone-based calibration"""
        # Update config with current values
        self.analyzer.config['calibration']['sensor_width_mm'] = self.sensor_width_input.value()
        self.analyzer.config['calibration']['focal_length_mm'] = self.focal_length_input.value()
        self.analyzer.config['calibration']['altitude_m'] = self.altitude_input.value()
        self.analyzer.config['calibration']['tilt_angle_deg'] = self.tilt_angle_input.value()
        
        # Calculate calibration factor
        meters_per_pixel = self.calibrator.calculate_drone_calibration()
        
        self.calibration_result = CalibrationResult(meters_per_pixel=meters_per_pixel)
        self.status_label.setText(f"Drone calibration complete: {meters_per_pixel:.6f} m/px")
        
        # Emit signal
        self.calibration_complete.emit(self.calibration_result)
    
    def prompt_for_calibration_distance(self):
        """Prompt user for calibration distance before manual calibration"""
        # Use the value from the input field
        self.calibration_distance = self.known_distance_input.value()
        
        # Update config
        self.analyzer.config['calibration']['known_distance_m'] = self.calibration_distance
        
        # Show confirmation dialog
        distance_msg = QMessageBox()
        distance_msg.setWindowTitle("Calibration Distance")
        distance_msg.setText(f"Using {self.calibration_distance} meters as the known distance.")
        distance_msg.setInformativeText("Click OK to continue with manual calibration. You will select two points in the next window.")
        distance_msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        return distance_msg.exec() == QMessageBox.Ok
    
    def start_manual_calibration(self):
        """Switch to OpenCV window for manual calibration"""
        # Prompt for calibration distance first
        if not self.prompt_for_calibration_distance():
            return
        
        # Start calibration in OpenCV window
        self.status_label.setText(f"Manual calibration started with {self.calibration_distance}m distance. Please select two points in the OpenCV window.")
        
        # Perform calibration
        self.calibration_result = self.calibrator.perform_manual_calibration(self.calibration_frame, self.calibration_distance)
        
        if self.calibration_result:
            self.status_label.setText(f"Manual calibration complete: {self.calibration_result.meters_per_pixel:.6f} m/px")
            
            # Emit signal
            self.calibration_complete.emit(self.calibration_result)
    
    def start_perspective_calibration(self):
        """Switch to OpenCV window for perspective calibration"""
        # Prompt for calibration distance first
        if not self.prompt_for_calibration_distance():
            return
        
        self.status_label.setText("Perspective calibration started. Please select four points in the OpenCV window.")
        
        # Perform calibration
        self.calibration_result = self.calibrator.setup_perspective_transform(self.calibration_frame, self.calibration_distance)
        
        if self.calibration_result:
            self.status_label.setText(f"Perspective calibration complete: {self.calibration_result.meters_per_pixel:.6f} m/px")
            
            # Emit signal
            self.calibration_complete.emit(self.calibration_result)


class TrafficAnalyzerGUI(QMainWindow):
    """Main GUI application for traffic analysis"""
    
    def __init__(self, config_path=None):
        super().__init__()
        
        # Initialize analyzer
        self.analyzer = TrafficAnalyzer(config_path=config_path, gui_mode=True)
        self.analyzer.setup()
        
        # Set up GUI
        self.setWindowTitle("Traffic Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Threading
        self.video_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Toolbar with actions
        self.init_toolbar()
        
        # Create tab widget for different views
        tabs = QTabWidget()
        
        # Main visualization tab
        visualization_tab = QWidget()
        viz_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: #222;")
        viz_layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Play/pause button
        self.play_pause_btn = QPushButton("Start")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_btn)
        
        # Visualization toggles
        self.show_detections_cb = QCheckBox("Show Detections")
        self.show_detections_cb.setChecked(True)
        self.show_detections_cb.stateChanged.connect(self.toggle_detections)
        controls_layout.addWidget(self.show_detections_cb)
        
        self.show_tracks_cb = QCheckBox("Show Tracks")
        self.show_tracks_cb.setChecked(True)
        self.show_tracks_cb.stateChanged.connect(self.toggle_tracks)
        controls_layout.addWidget(self.show_tracks_cb)
        
        self.show_info_cb = QCheckBox("Show Info")
        self.show_info_cb.setChecked(True)
        self.show_info_cb.stateChanged.connect(self.toggle_info)
        controls_layout.addWidget(self.show_info_cb)
        
        self.show_speed_chart_cb = QCheckBox("Speed Chart")
        self.show_speed_chart_cb.setChecked(False)
        self.show_speed_chart_cb.stateChanged.connect(self.toggle_speed_chart)
        controls_layout.addWidget(self.show_speed_chart_cb)
        
        self.show_class_chart_cb = QCheckBox("Class Chart")
        self.show_class_chart_cb.setChecked(False)
        self.show_class_chart_cb.stateChanged.connect(self.toggle_class_chart)
        controls_layout.addWidget(self.show_class_chart_cb)
        
        viz_layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        viz_layout.addWidget(self.progress_bar)
        
        visualization_tab.setLayout(viz_layout)
        tabs.addTab(visualization_tab, "Video Analysis")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # Video settings
        video_group = QGroupBox("Video Settings")
        video_layout = QVBoxLayout()
        
        # Video file selection
        video_file_layout = QHBoxLayout()
        video_file_label = QLabel("Video File:")
        self.video_file_input = QLabel(self.analyzer.config['video']['default_path'])
        self.video_file_input.setStyleSheet("font-weight: bold;")
        select_video_btn = QPushButton("Browse...")
        select_video_btn.clicked.connect(self.select_video_file)
        
        video_file_layout.addWidget(video_file_label)
        video_file_layout.addWidget(self.video_file_input)
        video_file_layout.addWidget(select_video_btn)
        video_layout.addLayout(video_file_layout)
        
        # FPS control
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS Override:")
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 60)
        self.fps_input.setValue(self.analyzer.config['video'].get('fps', 30))
        
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_input)
        video_layout.addLayout(fps_layout)
        
        video_group.setLayout(video_layout)
        settings_layout.addWidget(video_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()
        
        # Model file selection
        model_file_layout = QHBoxLayout()
        model_file_label = QLabel("Model File:")
        self.model_file_input = QLabel(self.analyzer.config['model']['path'])
        self.model_file_input.setStyleSheet("font-weight: bold;")
        select_model_btn = QPushButton("Browse...")
        select_model_btn.clicked.connect(self.select_model_file)
        
        model_file_layout.addWidget(model_file_label)
        model_file_layout.addWidget(self.model_file_input)
        model_file_layout.addWidget(select_model_btn)
        detection_layout.addLayout(model_file_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.analyzer.config['model']['confidence'] * 100))
        self.conf_value_label = QLabel(f"{self.analyzer.config['model']['confidence']}")
        self.conf_slider.valueChanged.connect(self.update_conf_value)
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        detection_layout.addLayout(conf_layout)
        
        detection_group.setLayout(detection_layout)
        settings_layout.addWidget(detection_group)
        
        # Tracking settings
        tracking_group = QGroupBox("Tracking Settings")
        tracking_layout = QVBoxLayout()
        
        # Distance threshold
        dist_layout = QHBoxLayout()
        dist_label = QLabel("Distance Threshold:")
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(10, 200)
        self.dist_slider.setValue(self.analyzer.config['tracking']['distance_threshold'])
        self.dist_value_label = QLabel(f"{self.analyzer.config['tracking']['distance_threshold']}")
        self.dist_slider.valueChanged.connect(self.update_dist_value)
        
        dist_layout.addWidget(dist_label)
        dist_layout.addWidget(self.dist_slider)
        dist_layout.addWidget(self.dist_value_label)
        tracking_layout.addLayout(dist_layout)
        
        # Smoothing alpha
        smooth_layout = QHBoxLayout()
        smooth_label = QLabel("Smoothing Factor:")
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(1, 99)
        self.smooth_slider.setValue(int(self.analyzer.config['tracking']['smoothing_alpha'] * 100))
        self.smooth_value_label = QLabel(f"{self.analyzer.config['tracking']['smoothing_alpha']}")
        self.smooth_slider.valueChanged.connect(self.update_smooth_value)
        
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(self.smooth_slider)
        smooth_layout.addWidget(self.smooth_value_label)
        tracking_layout.addLayout(smooth_layout)
        
        tracking_group.setLayout(tracking_layout)
        settings_layout.addWidget(tracking_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Report file
        report_file_layout = QHBoxLayout()
        report_file_label = QLabel("Report File:")
        self.report_file_input = QLabel(self.analyzer.config['output']['report_path'])
        self.report_file_input.setStyleSheet("font-weight: bold;")
        select_report_btn = QPushButton("Browse...")
        select_report_btn.clicked.connect(self.select_report_file)
        
        report_file_layout.addWidget(report_file_label)
        report_file_layout.addWidget(self.report_file_input)
        report_file_layout.addWidget(select_report_btn)
        output_layout.addLayout(report_file_layout)
        
        # Save video option
        save_video_layout = QHBoxLayout()
        self.save_video_cb = QCheckBox("Save Output Video")
        self.save_video_cb.setChecked(self.analyzer.config['output'].get('save_video', False))
        
        output_video_label = QLabel("Output Video File:")
        self.output_video_input = QLabel(self.analyzer.config['output'].get('output_video_path', 'output_video.mp4'))
        select_output_video_btn = QPushButton("Browse...")
        select_output_video_btn.clicked.connect(self.select_output_video_file)
        
        save_video_layout.addWidget(self.save_video_cb)
        save_video_layout.addWidget(output_video_label)
        save_video_layout.addWidget(self.output_video_input)
        save_video_layout.addWidget(select_output_video_btn)
        output_layout.addLayout(save_video_layout)
        
        output_group.setLayout(output_layout)
        settings_layout.addWidget(output_group)
        
        # Calibration button
        self.calibration_btn = QPushButton("Camera Calibration")
        self.calibration_btn.clicked.connect(self.open_calibration_dialog)
        settings_layout.addWidget(self.calibration_btn)
        
        # Apply settings button
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        settings_layout.addWidget(self.apply_settings_btn)
        
        settings_tab.setLayout(settings_layout)
        tabs.addTab(settings_tab, "Settings")
        
        # Add tabs to main layout
        main_layout.addWidget(tabs)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Set main widget and layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def init_toolbar(self):
        """Initialize toolbar with actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Open action
        open_action = QAction("Open", self)
        open_action.setStatusTip("Open video file")
        open_action.triggered.connect(self.select_video_file)
        toolbar.addAction(open_action)
        
        # Start action
        self.start_action = QAction("Start", self)
        self.start_action.setStatusTip("Start processing")
        self.start_action.triggered.connect(self.start_processing)
        toolbar.addAction(self.start_action)
        
        # Stop action
        self.stop_action = QAction("Stop", self)
        self.stop_action.setStatusTip("Stop processing")
        self.stop_action.triggered.connect(self.stop_processing)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        toolbar.addSeparator()
        
        # Calibration action
        calibration_action = QAction("Calibration", self)
        calibration_action.setStatusTip("Camera calibration")
        calibration_action.triggered.connect(self.open_calibration_dialog)
        toolbar.addAction(calibration_action)
        
        # Report action
        report_action = QAction("Generate Report", self)
        report_action.setStatusTip("Generate traffic report")
        report_action.triggered.connect(self.generate_report)
        toolbar.addAction(report_action)
    
    def select_video_file(self):
        """Open file dialog to select video file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options
        )
        
        if file_name:
            self.analyzer.config['video']['default_path'] = file_name
            self.video_file_input.setText(file_name)
            self.statusBar.showMessage(f"Selected video: {file_name}")
    
    def select_model_file(self):
        """Open file dialog to select model file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pt *.pth);;All Files (*)", options=options
        )
        
        if file_name:
            self.analyzer.config['model']['path'] = file_name
            self.model_file_input.setText(file_name)
            self.statusBar.showMessage(f"Selected model: {file_name}")
    
    def select_report_file(self):
        """Open file dialog to select report file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select Report File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        
        if file_name:
            self.analyzer.config['output']['report_path'] = file_name
            self.report_file_input.setText(file_name)
            self.statusBar.showMessage(f"Report will be saved to: {file_name}")

    def select_output_video_file(self):
        """Open file dialog to select output video file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select Output Video File", "", "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)", options=options
        )
        
        if file_name:
            self.analyzer.config['output']['output_video_path'] = file_name
            self.output_video_input.setText(file_name)
            self.statusBar.showMessage(f"Output video will be saved to: {file_name}")

    def update_conf_value(self):
        """Update confidence threshold value from slider"""
        value = self.conf_slider.value() / 100.0
        self.conf_value_label.setText(f"{value:.2f}")
        self.analyzer.config['model']['confidence'] = value

    def update_dist_value(self):
        """Update distance threshold value from slider"""
        value = self.dist_slider.value()
        self.dist_value_label.setText(f"{value}")
        self.analyzer.config['tracking']['distance_threshold'] = value

    def update_smooth_value(self):
        """Update smoothing factor value from slider"""
        value = self.smooth_slider.value() / 100.0
        self.smooth_value_label.setText(f"{value:.2f}")
        self.analyzer.config['tracking']['smoothing_alpha'] = value

    def apply_settings(self):
        """Apply settings to analyzer"""
        # Update FPS
        self.analyzer.config['video']['fps'] = self.fps_input.value()
        
        # Update save video option
        self.analyzer.config['output']['save_video'] = self.save_video_cb.isChecked()
        
        # Update model confidence threshold
        self.analyzer.config['model']['confidence'] = self.conf_slider.value() / 100.0
        
        # Update tracking parameters
        self.analyzer.config['tracking']['distance_threshold'] = self.dist_slider.value()
        self.analyzer.config['tracking']['smoothing_alpha'] = self.smooth_slider.value() / 100.0
        
        # Reinitialize detector with new settings
        try:
            self.analyzer.detector = VehicleDetector(self.analyzer.config)
            self.statusBar.showMessage("Settings applied successfully")
            
            # Show confirmation message
            QMessageBox.information(self, "Settings Applied", "The settings have been successfully applied.")
        except Exception as e:
            self.statusBar.showMessage(f"Error applying settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")

    def open_calibration_dialog(self):
        """Open calibration dialog"""
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Please stop processing before calibration.")
            return
            
        dialog = CalibrationDialog(self.analyzer, self)
        dialog.calibration_complete.connect(self.on_calibration_complete)
        dialog.show()

    def on_calibration_complete(self, calibration_result):
        """Handle calibration completion"""
        # Initialize tracker with calibration result
        self.analyzer.tracker = VehicleTracker(self.analyzer.config, calibration_result)
        self.statusBar.showMessage(f"Calibration complete: {calibration_result.meters_per_pixel:.6f} m/px")

    def start_processing(self):
        """Start video processing"""
        if self.video_thread and self.video_thread.isRunning():
            # Already running, toggle pause/resume
            self.toggle_play_pause()
            return
            
        # Check if calibration is done
        if not hasattr(self.analyzer, 'tracker') or self.analyzer.tracker is None:
            # Try to use drone calibration as default
            calibration_result = CalibrationResult(
                meters_per_pixel=self.analyzer.calibrator.calculate_drone_calibration()
            )
            self.analyzer.tracker = VehicleTracker(self.analyzer.config, calibration_result)
            self.statusBar.showMessage(f"Using default calibration: {calibration_result.meters_per_pixel:.6f} m/px")
        
        # Create and start video processing thread
        self.video_thread = VideoThread(self.analyzer)
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.processing_finished.connect(self.processing_finished)
        self.video_thread.progress_update.connect(self.progress_bar.setValue)
        self.video_thread.status_update.connect(self.statusBar.showMessage)
        self.video_thread.error_occurred.connect(self.handle_error)
        
        self.video_thread.start()
        
        # Update UI
        self.play_pause_btn.setText("Pause")
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)

    def stop_processing(self):
        """Stop video processing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.statusBar.showMessage("Processing stopped")
            
            # Update UI
            self.play_pause_btn.setText("Start")
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def toggle_play_pause(self):
        """Toggle play/pause state"""
        if not self.video_thread or not self.video_thread.isRunning():
            self.start_processing()
            return
            
        if self.video_thread.is_paused:
            self.video_thread.resume()
            self.play_pause_btn.setText("Pause")
            self.statusBar.showMessage("Processing resumed")
        else:
            self.video_thread.pause()
            self.play_pause_btn.setText("Resume")
            self.statusBar.showMessage("Processing paused")

    def toggle_detections(self, state):
        """Toggle detection visualization"""
        if self.analyzer and self.analyzer.visualizer:
            self.analyzer.visualizer.show_detections = bool(state)

    def toggle_tracks(self, state):
        """Toggle tracks visualization"""
        if self.analyzer and self.analyzer.visualizer:
            self.analyzer.visualizer.show_tracks = bool(state)

    def toggle_info(self, state):
        """Toggle information overlay"""
        if self.analyzer and self.analyzer.visualizer:
            self.analyzer.visualizer.show_info = bool(state)

    def toggle_speed_chart(self, state):
        """Toggle speed chart visualization"""
        if self.analyzer and self.analyzer.visualizer:
            self.analyzer.visualizer.show_speed_chart = bool(state)

    def toggle_class_chart(self, state):
        """Toggle class chart visualization"""
        if self.analyzer and self.analyzer.visualizer:
            self.analyzer.visualizer.show_class_chart = bool(state)

    def update_frame(self, frame, tracking_results, detections):
        """Update displayed frame"""
        # Convert OpenCV BGR image to RGB for Qt
        h, w, ch = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from the frame data
        image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        
        # Display the image
        pixmap = QPixmap.fromImage(image)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)

    def processing_finished(self):
        """Handle processing finished event"""
        self.statusBar.showMessage("Processing complete")
        
        # Update UI
        self.play_pause_btn.setText("Start")
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        
        # Show completion message
        QMessageBox.information(self, "Processing Complete", 
                              f"Video processing complete.\nReport saved to: {self.analyzer.config['output']['report_path']}")

    def handle_error(self, error_message):
        """Handle error from processing thread"""
        self.statusBar.showMessage(f"Error: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Error", f"An error occurred during processing:\n{error_message}")
        
        # Reset UI
        self.play_pause_btn.setText("Start")
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)

    def generate_report(self):
        """Generate traffic analysis report"""
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Please wait for processing to complete before generating report.")
            return
            
        if not hasattr(self.analyzer, 'tracker') or self.analyzer.tracker is None:
            QMessageBox.warning(self, "Warning", "No tracking data available. Please process a video first.")
            return
            
        # Generate report
        try:
            df = self.analyzer.visualizer.generate_report(
                {'speeds': self.analyzer.visualizer.vehicle_speeds},
                self.analyzer.config['output']['report_path']
            )
            
            if df is not None:
                QMessageBox.information(self, "Report Generated", 
                                     f"Report saved to: {self.analyzer.config['output']['report_path']}\n"
                                     f"Total vehicles: {len(df)}")
            else:
                QMessageBox.warning(self, "No Data", "No vehicle data available for report.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_thread and self.video_thread.isRunning():
            # Ask for confirmation
            reply = QMessageBox.question(self, "Confirm Exit", 
                                      "Processing is still running. Are you sure you want to exit?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.video_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()