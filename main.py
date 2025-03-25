# main.py - Main script to launch the Traffic Analyzer application

import sys
import os
import argparse
import torch
from PyQt5.QtWidgets import QApplication

# Import our GUI application
from traffic_analyzer_gui import TrafficAnalyzerGUI

def check_gpu():
    """Check for GPU availability and print info"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        print(f"[GPU] CUDA device detected: {device_name}")
        print(f"[GPU] CUDA version: {cuda_version}")
        print(f"[GPU] Device count: {torch.cuda.device_count()}")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        return True
    else:
        print("[GPU] No CUDA device detected. Processing will be CPU-only.")
        return False

def main():
    # Check for GPU
    has_gpu = check_gpu()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Traffic Analysis System")
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode (no GUI)')
    parser.add_argument('--video', '-v', type=str, help='Path to video file')
    parser.add_argument('--model', '-m', type=str, help='Path to YOLO model file')
    parser.add_argument('--output', '-o', type=str, help='Path to output report file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (ignore GPU)')
    
    args = parser.parse_args()
    
    # If CPU is forced, disable CUDA
    if args.cpu and has_gpu:
        print("[GPU] Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # If CLI mode is selected, run the command-line version
    if args.cli:
        from traffic_analyzer import main as cli_main
        cli_main()
        return
    
    # Otherwise, launch the GUI application
    app = QApplication(sys.argv)
    window = TrafficAnalyzerGUI(config_path=args.config)
    
    # Apply command-line arguments if provided
    if args.video:
        window.analyzer.config['video']['default_path'] = args.video
        window.video_file_input.setText(args.video)
    
    if args.model:
        window.analyzer.config['model']['path'] = args.model
        window.model_file_input.setText(args.model)
    
    if args.output:
        window.analyzer.config['output']['report_path'] = args.output
        window.report_file_input.setText(args.output)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
