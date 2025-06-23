#!/usr/bin/env python3
"""
Simple runner script for the Reel Maker application
Run this file to start the application
"""

import sys
import os
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
from main import main

if __name__ == "__main__":
    print("🎬 Starting Automated Reels Video Maker...")
    print("📝 Make sure you have installed all requirements: pip install -r requirements.txt")
    print("🌐 The app will open in your browser automatically")
    print("-" * 50)
    
    main()