# 🎬 Professional Reel Maker with Reddit Integration

Create engaging vertical videos with Reddit-style posts, AI-powered transcription, and professional text overlays. Features dual audio system, dynamic font sizing, dead air support, and instant text display.

## ✨ Features

- 🔥 **Reddit Post Generation** - Create authentic Reddit-style posts with custom content
- 🎵 **Dual Audio System** - Title audio + Main story audio with precise timing
- 🤖 **AI Transcription** - GPU-accelerated Whisper transcription
- 📱 **9:16 Aspect Ratio** - Perfect for TikTok, Instagram Reels, YouTube Shorts
- ⚡ **Instant Text Display** - No fade effects, text appears immediately
- 🔤 **Dynamic Font Sizing** - Automatic font adjustment for 1-5 words per segment
- 📐 **Permanent Margins** - 10% margins maintained to prevent text overflow
- ⏱️ **Dead Air Support** - 3.5s ending silence with last text remaining visible
- 🎨 **Live Preview** - Real-time preview that matches final output exactly
- 🌅 **Background Effects** - Vignette and heartbeat animation
- 📝 **Script Matching** - Import your script for improved text accuracy

## 🔧 System Requirements

### **Required Software:**
- **Python 3.11.9** (recommended version for best compatibility)
- **CUDA 11.8** (for GPU acceleration)
- **FFmpeg** (for video/audio processing)

### **Hardware Requirements:**
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for Whisper transcription)
- **RAM**: 8GB+ system RAM
- **Storage**: 2GB+ free space for temporary processing

## 📦 Installation Guide

### Step 1: Install Python 3.11.9

**Windows:**
1. Download Python 3.11.9 from [python.org](https://www.python.org/downloads/release/python-3119/)
2. During installation, **check "Add Python to PATH"**
3. Verify: `python --version` should show `Python 3.11.9`

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Mac:**
```bash
brew install python@3.11
```

### Step 2: Install CUDA 11.8

**Windows & Linux:**
1. Download CUDA 11.8 from [NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Follow NVIDIA's installation guide for your OS
3. Verify: `nvcc --version` should show `release 11.8`

**Important:** Ensure your NVIDIA GPU driver supports CUDA 11.8

### Step 3: Install FFmpeg

#### **Windows:**
1. **Download FFmpeg:**
   - Go to [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Click "Windows" → "Windows builds by BtbN"
   - Download the latest release (e.g., `ffmpeg-master-latest-win64-gpl.zip`)

2. **Extract and Setup:**
   - Extract to `C:\ffmpeg\` (or your preferred location)
   - The folder structure should be: `C:\ffmpeg\bin\ffmpeg.exe`

3. **Add to PATH:**
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click "OK" on all dialogs
   - **Restart your command prompt/terminal**

4. **Verify Installation:**
   ```cmd
   ffmpeg -version
   ```
   Should display FFmpeg version info.

#### **Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### **Mac:**
```bash
brew install ffmpeg
```

### Step 4: Install Python Dependencies

1. **Clone/Download the project files**

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv reel_maker_env
   
   # Windows:
   reel_maker_env\Scripts\activate
   
   # Linux/Mac:
   source reel_maker_env/bin/activate
   ```

3. **Install PyTorch with CUDA 11.8:**
   ```bash
   pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start

1. **Run the application:**
   ```bash
   python run.py
   ```

2. **Open your browser to:** `http://localhost:7860`

### Step-by-Step Workflow

#### **Tab 1: Reddit Post Generator**
1. Enter your post title
2. Optionally customize username and avatar
3. Set likes and comments count
4. Click "Generate Reddit Post"
5. **Save the generated image** (right-click → save)

#### **Tab 2: Generate Reddit Reel**
1. **Upload title audio** (plays during Reddit post display)
2. **Upload main audio** (your story content)
3. **Upload background video** (any aspect ratio, will be cropped to 9:16)
4. **Optional:** Paste your original script for better text accuracy
5. Click "Create Reddit Reel"

#### **Tab 3: Style & Preview**
- Customize font, colors, position, and effects
- Adjust words per segment (1-5) - font auto-adjusts
- Real-time preview shows exactly how your video will look

## ⚙️ Advanced Configuration

### Audio Requirements
- **Title Audio:** 0.5s - 30s (displays Reddit post)
- **Main Audio:** 1s - 5 minutes (displays synchronized text)
- **Supported formats:** MP3, WAV, M4A

### Video Requirements
- **Background Video:** MP4 recommended
- **Any aspect ratio** (will be auto-cropped to 9:16)
- **Duration:** Should be longer than your audio

### Text Styling
- **Dynamic Font Sizing:** Automatically adjusts based on word count
- **Permanent Margins:** 10% margins prevent text overflow
- **Word Segments:** 1-5 words per segment with optimal sizing

## 🔧 Troubleshooting

### Common Issues

**❌ "CUDA not available"**
- Verify CUDA 11.8 installation: `nvcc --version`
- Check GPU compatibility
- Reinstall PyTorch with CUDA: `pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

**❌ "FFmpeg not found"**
- Verify FFmpeg installation: `ffmpeg -version`
- Ensure FFmpeg bin folder is in PATH
- Restart terminal after PATH changes

**❌ "Audio processing failed"**
- Check audio file format (use MP3/WAV)
- Ensure audio files are not corrupted
- Try shorter audio clips for testing

**❌ "Text overflowing screen"**
- Update to latest version (includes dynamic font sizing)
- Use 1-3 words per segment for longer text
- Check that permanent margins are enabled

**❌ "Reddit post not appearing"**
- Generate Reddit post BEFORE creating reel
- Ensure Reddit image is properly generated
- Check that title audio duration is reasonable (under 30s)

### Performance Tips

**🚀 GPU Acceleration:**
- Monitor GPU memory usage
- Close other GPU-intensive applications
- Use smaller Whisper model if needed: `model_size="tiny"`

**🎵 Audio Quality:**
- Use clear, high-quality audio
- Avoid background noise
- Normalize audio levels

**🎬 Video Quality:**
- Use high-resolution background videos
- Ensure good lighting in source video
- Test with shorter videos first

## 📝 File Structure

```
professional-reel-maker/
├── run.py                    # Main application launcher   
├── src/                  
   ├── audio_processor.py        # Audio transcription and processing
   ├── video_processor.py        # Video composition and effects
   ├── text_sync.py             # Text synchronization and timing
   ├── reddit_generator.py   
   ├── main.py                # Reddit post template generation
├── requirements.txt         # Python dependencies
├── image_src/               # Optional: Custom Reddit post icons
└── README.md               # This file
```

## 🎯 Output Specifications

**Final Video:**
- **Resolution:** 1080x1920 (9:16)
- **Frame Rate:** 30 FPS
- **Audio:** 128kbps AAC
- **Format:** MP4
- **Features:** 
  - Reddit post display during title
  - Synchronized text during story
  - 3.5s dead air at end
  - Professional text styling
  - Background effects

## 🆘 Support

**System Check:**
Run the application with `python run.py` to see a comprehensive system check including:
- Python version verification
- CUDA availability
- FFmpeg installation
- Required modules
- GPU memory status

**Debug Mode:**
The application includes detailed logging. Check console output for specific error messages and troubleshooting hints.

## 📄 License

This project is designed for creating engaging social media content. Please ensure compliance with platform guidelines and copyright laws when using generated content.

---

**🎬 Ready to create amazing reels? Run `python run.py` and start making professional vertical videos!**