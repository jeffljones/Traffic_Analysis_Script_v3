# config.py - Configuration settings

import yaml
import os

def load_config(config_path=None):
    """Load configuration from YAML file or use defaults"""
    default_config = {
        "model": {
            "path": "yolov8n.pt",
            "confidence": 0.3
        },
        "video": {
            "default_path": "traffic_video_1min.mp4",
            "fps": 30
        },
        "calibration": {
            "sensor_width_mm": 13.2,
            "focal_length_mm": 8.8,
            "image_width_px": 4000,
            "altitude_m": 50,
            "tilt_angle_deg": 30,
            "known_distance_m": 10.0
        },
        "tracking": {
            "distance_threshold": 80,
            "smoothing_alpha": 0.3,
            "removal_threshold": 2.0,
            "boundary_percent": 15,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3
        },
        "output": {
            "report_path": "traffic_report.csv",
            "save_video": False,
            "output_video_path": "output_video.mp4"
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge user config with default config
                for category in user_config:
                    if category in default_config:
                        default_config[category].update(user_config[category])
                    else:
                        default_config[category] = user_config[category]
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            
    return default_config
