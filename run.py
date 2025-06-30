#!/usr/bin/env python3
"""
Enhanced runner script for the Professional Reel Maker with Reddit Integration
Run this file to start the application with all new features
"""

import sys
import os
import torch
import logging
import subprocess

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements and GPU availability"""
    print("🔍 Checking system requirements...")
    
    # GPU Check
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA version: {torch.version.cuda}")
        print(f"🎮 GPU device: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU memory: {gpu_memory:.1f}GB")
    else:
        print("⚠️  No GPU detected - using CPU (slower transcription)")
    
    # Python version check
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        return False
    
    return True

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            print(f"🎬 FFmpeg: {version_line}")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found - audio processing will be limited")
        print("💡 Install FFmpeg: https://ffmpeg.org/download.html")
        return False

def check_required_modules():
    """Check if all required modules are available"""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('gradio', 'gradio'),
        ('whisper', 'openai-whisper'),
        ('soundfile', 'soundfile'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('librosa', 'librosa'),
        ('ffmpeg', 'ffmpeg-python')
    ]
    
    missing_modules = []
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_modules.append(package_name)
    
    if missing_modules:
        print(f"\n💡 Install missing modules:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True

def create_image_src_directory():
    """Create image_src directory with README if it doesn't exist"""
    image_src_path = os.path.join(os.path.dirname(__file__), 'image_src')
    
    if not os.path.exists(image_src_path):
        os.makedirs(image_src_path)
        print(f"📁 Created image_src directory: {image_src_path}")
        
        # Create README file
        readme_content = """# Image Assets for Reddit Generator

This directory contains PNG assets used by the Reddit post generator.

## Required Files (optional - fallbacks will be used if missing):
- heart.png - Heart icon for likes
- share.png - Share icon 
- medal.png - Medal badge
- trophy.png - Trophy badge
- radioactive.png - Radioactive badge
- mask.png - Mask badge
- handshake.png - Handshake badge
- rocket.png - Rocket badge
- gem.png - Gem badge
- fire.png - Fire badge

## Notes:
- All icons should be square (e.g., 64x64px)
- PNG format with transparency support
- If files are missing, colored fallback icons will be used
- Icons will be automatically resized to fit the Reddit post template
"""
        
        readme_path = os.path.join(image_src_path, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print("📝 Created README.md in image_src directory")
    else:
        print(f"📁 image_src directory already exists: {image_src_path}")

def print_startup_banner():
    """Print enhanced startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🎬 PROFESSIONAL REEL MAKER + REDDIT INTEGRATION                ║
║                                                                              ║
║  🚀 Features:                                                                ║
║    • Dual Audio System (Title + Main Story)                                 ║
║    • Reddit Post Generation & Integration                                    ║
║    • AI-Powered Transcription (Whisper)                                     ║
║    • Live Preview with Instant Display                                      ║
║    • Professional Text Overlays                                             ║
║    • 9:16 Vertical Video Output                                             ║
║    • GPU Acceleration Support                                               ║
║                                                                              ║
║  🎵 Audio System:                                                            ║
║    • Title audio displays Reddit post                                       ║
║    • 1-second delay between title and main                                  ║
║    • Main audio displays synchronized text                                  ║
║                                                                              ║
║  💡 Workflow:                                                                ║
║    1. Generate Reddit post in Tab 1                                         ║
║    2. Upload title + main audio + video in Tab 2                            ║
║    3. Customize styling in Tab 3                                            ║
║    4. Process and download your reel!                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Enhanced main entry point with comprehensive system checks"""
    print_startup_banner()
    
    print("\n🔧 System Requirements Check:")
    print("=" * 50)
    
    # Check Python and GPU
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        input("Press Enter to exit...")
        return
    
    print("\n📦 Module Availability Check:")
    print("=" * 50)
    
    # Check required modules
    if not check_required_modules():
        print("\n❌ Missing required modules. Please install them and try again.")
        input("Press Enter to exit...")
        return
    
    print("\n🛠️ External Dependencies:")
    print("=" * 50)
    
    # Check FFmpeg (optional but recommended)
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("⚠️  FFmpeg missing - some audio features may be limited")
        continue_anyway = input("Continue anyway? (y/N): ").lower().strip()
        if continue_anyway != 'y':
            print("Please install FFmpeg and try again.")
            return
    
    print("\n📁 Directory Setup:")
    print("=" * 50)
    
    # Create necessary directories
    create_image_src_directory()
    
    print("\n🚀 Starting Application:")
    print("=" * 50)
    
    # Add src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"📂 Added src directory to path: {src_path}")
    else:
        # If no src directory, assume files are in current directory
        sys.path.insert(0, os.path.dirname(__file__))
        print(f"📂 Using current directory: {os.path.dirname(__file__)}")
    
    try:
        # Import and run the main application
        print("📥 Loading application modules...")
        from main import main as app_main
        
        print("✅ All modules loaded successfully")
        print("🌐 Starting web interface...")
        print("\n" + "=" * 70)
        print("🎉 APPLICATION READY!")
        print("📱 The app will open in your browser automatically")
        print("🔗 Manual access: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the application")
        print("=" * 70)
        
        # Start the application
        app_main()
        
    except ImportError as e:
        print(f"\n❌ Failed to import application modules: {e}")
        print("💡 Make sure all files are in the correct location:")
        print("   - main.py")
        print("   - audio_processor.py") 
        print("   - video_processor.py")
        print("   - text_sync.py")
        print("   - reddit_generator.py")
        input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        print("👋 Thank you for using Professional Reel Maker!")
        
    except Exception as e:
        print(f"\n❌ Application failed to start: {e}")
        logger.exception("Application startup failed")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()