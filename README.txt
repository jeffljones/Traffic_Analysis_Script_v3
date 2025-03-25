# Traffic Analyzer System

A comprehensive system for analyzing traffic footage, detecting vehicles, tracking their movement, measuring speeds, and generating reports.

## Overview

The Traffic Analyzer is a Python-based application that uses computer vision and machine learning to process traffic video footage. The system detects vehicles, tracks them across frames, estimates their speeds, and generates detailed reports. It features both a graphical user interface (GUI) for interactive use and a command-line interface for batch processing.

## Features

- Vehicle detection using YOLO
- Robust tracking with SORT algorithm and Kalman filtering
- Multiple calibration methods for accurate speed measurement
- Real-time visualization of detection and tracking results
- Detailed reporting of vehicle counts, classifications, and speeds
- Interactive user interface with customizable settings
- Exportable CSV reports and video outputs

## Directory Structure and Files

The system consists of the following files, which should be organized as shown:

```
traffic_analyzer/
│
├── main.py                  # Main entry point for the application
├── traffic_analyzer.py      # Core traffic analysis functionality
├── traffic_analyzer_gui.py  # GUI implementation
├── config.py               # Configuration management
├── calibration.py          # Camera calibration functionality
├── detection.py            # Vehicle detection using YOLO
├── tracker.py              # Vehicle tracking implementation
├── sort.py                 # SORT algorithm implementation
├── visualization.py        # Visualization utilities
│
├── config.yml              # Default configuration (created on first run)
│
├── models/                 # Directory for YOLO model files
│   └── best.pt             # YOLO model file (not included, must be provided)
│
├── output/                 # Default directory for results
│   ├── traffic_report.csv  # Generated CSV reports
│   └── output_video.mp4    # Processed video output
│
└── README.md               # Documentation
```

## Installation

1. Clone or download the repository:
   ```
   git clone https://github.com/yourusername/traffic_analyzer.git
   cd traffic_analyzer
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download a YOLO model file (`.pt`) trained for vehicle detection and place it in the `models` directory.

## Dependencies

- Python 3.8+
- OpenCV 4.5+
- NumPy
- PyQt5
- ultralytics (YOLO)
- FilterPy (for Kalman filtering)
- pandas
- matplotlib
- scipy

## Usage

### GUI Mode

1. Launch the application:
   ```
   python main.py
   ```

2. In the GUI:
   - Go to the "Settings" tab to configure video, model, and output options
   - Perform camera calibration using the "Calibration" button
   - Return to the "Video Analysis" tab and click "Start" to begin processing
   - Use the checkboxes to toggle visualization options
   - Click "Generate Report" to save the results

### Command-Line Mode

For batch processing, use the command-line interface:

```
python main.py --cli --video path/to/traffic_video.mp4 --model path/to/model.pt --output path/to/report.csv
```

Available command-line arguments:
- `--cli`: Run in command-line mode (no GUI)
- `--video` or `-v`: Path to the video file
- `--model` or `-m`: Path to the YOLO model file
- `--output` or `-o`: Path to the output report file
- `--config` or `-c`: Path to a custom configuration file

## Calibration Methods

The system offers three calibration methods:

1. **Drone-based calibration**: Uses drone camera parameters, altitude, and tilt angle
2. **Manual calibration**: Select two points of known real-world distance
3. **Perspective calibration**: Set up a perspective transformation with four points

Calibration is essential for accurate speed measurements.

## Configuration

The system uses a configuration file (`config.yml`) with the following sections:

- **model**: YOLO model settings (path, confidence threshold)
- **video**: Video input settings (path, FPS)
- **calibration**: Camera calibration parameters
- **tracking**: Tracking algorithm parameters
- **output**: Report and video output settings

The configuration file is created with default values on the first run and can be edited directly or through the GUI.

## Outputs

The system generates the following outputs:

1. **CSV Report**: Contains detailed information about each detected vehicle:
   - Vehicle ID
   - Class (car, truck, motorcycle, etc.)
   - Speed (mph)
   - Timestamp
   - Direction of travel

2. **Processed Video** (optional): Video file with visualization overlays showing:
   - Detection bounding boxes
   - Tracking information
   - Speed measurements
   - Statistics and charts

## Troubleshooting

- **YOLO model errors**: Ensure you're using a compatible YOLO model file (v8 recommended)
- **Calibration issues**: For accurate speed measurements, proper calibration is essential
- **Performance problems**: Reduce resolution or use a lighter YOLO model for faster processing
- **Tracking inaccuracies**: Adjust tracking parameters in settings (distance threshold, smoothing factor)

## License

[Add appropriate license information here]

## Acknowledgments

- SORT algorithm by Alex Bewley et al.
- YOLO by Ultralytics