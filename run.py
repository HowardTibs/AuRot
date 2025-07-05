#!/usr/bin/env python3
"""
Enhanced runner script for the Professional Reel Maker with Reddit Integration
Run this file to start the application with all new features and consistency fixes
CONSISTENCY ENHANCED: Better system checks and cleanup support
"""

import sys
import os
import torch
import logging
import subprocess
import signal
import atexit
import gc
from pathlib import Path

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSystemChecker:
    """Enhanced system checker with consistency support"""
    
    def __init__(self):
        self.gpu_available = False
        self.ffmpeg_available = False
        self.python_version_ok = False
        self.modules_available = False
        
    def check_system_requirements(self):
        """Enhanced system requirements check"""
        print("🔍 Checking system requirements...")
        
        # Python version check
        python_version = sys.version_info
        print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("❌ Python 3.8+ required. Please upgrade Python.")
            self.python_version_ok = False
            return False
        else:
            print("✅ Python version compatible")
            self.python_version_ok = True
        
        # GPU Check with enhanced detection
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"🚀 CUDA version: {torch.version.cuda}")
                print(f"🎮 GPU device: {torch.cuda.get_device_name(0)}")
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"💾 GPU memory: {gpu_memory:.1f}GB")
                
                # Test GPU functionality
                test_tensor = torch.randn(10, 10).cuda()
                _ = test_tensor @ test_tensor.T
                print("✅ GPU functionality test passed")
                self.gpu_available = True
                
                # Clean up test tensor
                del test_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️  GPU available but not functional: {e}")
                self.gpu_available = False
        else:
            print("⚠️  No GPU detected - using CPU (slower transcription)")
            self.gpu_available = False
        
        return True
    
    def check_ffmpeg(self):
        """Enhanced FFmpeg check"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract version info
                version_line = result.stdout.split('\n')[0]
                print(f"🎬 FFmpeg: {version_line}")
                
                # Test FFmpeg functionality
                test_result = subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1', 
                                            '-f', 'null', '-'], 
                                           capture_output=True, text=True, timeout=15)
                if test_result.returncode == 0:
                    print("✅ FFmpeg functionality test passed")
                    self.ffmpeg_available = True
                    return True
                else:
                    print("⚠️  FFmpeg available but not fully functional")
                    self.ffmpeg_available = False
                    return False
            else:
                print("❌ FFmpeg not working properly")
                self.ffmpeg_available = False
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ FFmpeg not found - audio processing will be limited")
            print("💡 Install FFmpeg: https://ffmpeg.org/download.html")
            self.ffmpeg_available = False
            return False
    
    def check_required_modules(self):
        """Enhanced module availability check"""
        required_modules = [
            ('cv2', 'opencv-python'),
            ('gradio', 'gradio'),
            ('whisper', 'openai-whisper'),
            ('soundfile', 'soundfile'),
            ('numpy', 'numpy'),
            ('PIL', 'Pillow'),
            ('librosa', 'librosa'),
            ('ffmpeg', 'ffmpeg-python'),
            ('torch', 'torch'),
            ('torchaudio', 'torchaudio'),
        ]
        
        missing_modules = []
        available_modules = []
        
        for module_name, package_name in required_modules:
            try:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package_name} (v{version})")
                available_modules.append(package_name)
            except ImportError:
                print(f"❌ {package_name} - Missing")
                missing_modules.append(package_name)
        
        if missing_modules:
            print(f"\n💡 Install missing modules:")
            print(f"pip install {' '.join(missing_modules)}")
            self.modules_available = False
            return False
        else:
            print(f"\n✅ All {len(available_modules)} required modules available")
            self.modules_available = True
            return True
    
    def get_system_summary(self):
        """Get system summary for consistency tracking"""
        return {
            'python_ok': self.python_version_ok,
            'gpu_available': self.gpu_available,
            'ffmpeg_available': self.ffmpeg_available,
            'modules_available': self.modules_available,
            'system_ready': all([self.python_version_ok, self.modules_available])
        }

def setup_cleanup_handlers():
    """Setup cleanup handlers for graceful shutdown"""
    def cleanup_on_exit():
        """Cleanup function called on exit"""
        try:
            logger.info("🧹 Performing cleanup on exit...")
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear any OpenCV windows
            try:
                import cv2
                cv2.destroyAllWindows()
            except:
                pass
            
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def signal_handler(signum, frame):
        """Handle interrupt signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        cleanup_on_exit()
        sys.exit(0)
    
    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_image_src_directory():
    """Create image_src directory with README if it doesn't exist"""
    image_src_path = Path("image_src")
    
    if not image_src_path.exists():
        image_src_path.mkdir(exist_ok=True)
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

## Consistency Features:
- Asset cache is automatically cleared between generations
- Fallback icons ensure reliable operation
- Enhanced error handling prevents crashes
"""
        
        readme_path = image_src_path / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        
        print("📝 Created README.md in image_src directory")
    else:
        print(f"📁 image_src directory already exists: {image_src_path}")

def check_src_directory():
    """Check and setup src directory"""
    src_path = Path("src")
    current_dir = Path(".")
    
    # Check if we have source files in src/ or current directory
    main_files = ["main.py", "audio_processor.py", "video_processor.py", "text_sync.py", "reddit_generator.py"]
    
    src_files_exist = all((src_path / file).exists() for file in main_files)
    current_files_exist = all((current_dir / file).exists() for file in main_files)
    
    if src_files_exist:
        print(f"📂 Using src directory: {src_path.absolute()}")
        sys.path.insert(0, str(src_path.absolute()))
        return str(src_path.absolute())
    elif current_files_exist:
        print(f"📂 Using current directory: {current_dir.absolute()}")
        sys.path.insert(0, str(current_dir.absolute()))
        return str(current_dir.absolute())
    else:
        print("❌ Source files not found in src/ or current directory")
        print("💡 Required files:", ", ".join(main_files))
        return None

def print_startup_banner():
    """Print enhanced startup banner with consistency features"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🎬 PROFESSIONAL REEL MAKER + REDDIT INTEGRATION + CONSISTENCY        ║
║                                                                              ║
║  🚀 Features:                                                                ║
║    • Dual Audio System (Title + Main Story)                                 ║
║    • Reddit Post Generation & Integration                                    ║
║    • AI-Powered Transcription (Whisper)                                     ║
║    • Live Preview with Instant Display                                      ║
║    • Professional Text Overlays                                             ║
║    • 9:16 Vertical Video Output                                             ║
║    • GPU Acceleration Support                                               ║
║    • 3.5s Dead Air with Last Text Visible                                   ║
║    • Dynamic Font Sizing (1-5 words)                                        ║
║                                                                              ║
║  🧹 CONSISTENCY ENHANCEMENTS:                                                ║
║    • Enhanced cleanup between generations                                    ║
║    • Reliable text display across sessions                                  ║
║    • Improved state management                                              ║
║    • Better error handling and recovery                                     ║
║    • Automatic cache clearing                                               ║
║                                                                              ║
║  🎵 Audio System:                                                            ║
║    • Title audio displays Reddit post                                       ║
║    • 1-second delay between title and main                                  ║
║    • Main audio displays synchronized text                                  ║
║    • Perfect timing with dead air support                                   ║
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

def test_import_modules():
    """Test importing main application modules"""
    try:
        print("📥 Testing module imports...")
        
        # Test core modules
        from main import main as app_main
        print("✅ main.py imported successfully")
        
        from audio_processor import AudioProcessor
        print("✅ audio_processor.py imported successfully")
        
        from video_processor import ProVideoProcessor
        print("✅ video_processor.py imported successfully")
        
        from text_sync import TextSynchronizer
        print("✅ text_sync.py imported successfully")
        
        from reddit_generator import RedditTemplateGenerator
        print("✅ reddit_generator.py imported successfully")
        
        print("✅ All modules imported successfully")
        return True, app_main
        
    except ImportError as e:
        print(f"\n❌ Failed to import modules: {e}")
        print("💡 Make sure all files are in the correct location:")
        print("   - main.py")
        print("   - audio_processor.py") 
        print("   - video_processor.py")
        print("   - text_sync.py")
        print("   - reddit_generator.py")
        return False, None

def main():
    """Enhanced main entry point with comprehensive system checks and consistency support"""
    print_startup_banner()
    
    # Setup cleanup handlers
    setup_cleanup_handlers()
    
    print("\n🔧 System Requirements Check:")
    print("=" * 50)
    
    # Enhanced system checker
    checker = EnhancedSystemChecker()
    
    # Check Python and GPU
    if not checker.check_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        input("Press Enter to exit...")
        return
    
    print("\n📦 Module Availability Check:")
    print("=" * 50)
    
    # Check required modules
    if not checker.check_required_modules():
        print("\n❌ Missing required modules. Please install them and try again.")
        input("Press Enter to exit...")
        return
    
    print("\n🛠️ External Dependencies:")
    print("=" * 50)
    
    # Check FFmpeg (optional but recommended)
    ffmpeg_available = checker.check_ffmpeg()
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
    
    # Setup source directory
    src_dir = check_src_directory()
    if src_dir is None:
        print("\n❌ Source files not found. Please ensure all Python files are present.")
        input("Press Enter to exit...")
        return
    
    print("\n🧪 Module Import Test:")
    print("=" * 50)
    
    # Test module imports
    import_success, app_main = test_import_modules()
    if not import_success:
        input("Press Enter to exit...")
        return
    
    print("\n🚀 Starting Application:")
    print("=" * 50)
    
    # Get system summary
    system_summary = checker.get_system_summary()
    
    print("📊 System Summary:")
    for key, value in system_summary.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")
    
    if not system_summary['system_ready']:
        print("\n⚠️  System not fully ready, but attempting to start anyway...")
    
    try:
        print("\n✅ All checks passed successfully")
        print("🌐 Starting web interface...")
        print("\n" + "=" * 70)
        print("🎉 APPLICATION READY WITH CONSISTENCY ENHANCEMENTS!")
        print("📱 The app will open in your browser automatically")
        print("🔗 Manual access: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the application")
        print("🧹 Enhanced cleanup will run on shutdown")
        print("=" * 70)
        
        # Start the application
        app_main()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        print("🧹 Running enhanced cleanup...")
        print("👋 Thank you for using Professional Reel Maker!")
        
    except Exception as e:
        print(f"\n❌ Application failed to start: {e}")
        logger.exception("Application startup failed")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that all dependencies are installed")
        print("2. Ensure FFmpeg is properly installed")
        print("3. Verify GPU drivers if using CUDA")
        print("4. Try running with CPU only")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()