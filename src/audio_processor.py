"""
Audio processing module for Whisper AI transcription
WITH GPU ACCELERATION and optimizations for faster processing
"""

import whisper
import soundfile as sf
import tempfile
import logging
import threading
import torch
import warnings
import time
from typing import Dict, List, Optional

# Suppress specific Triton warnings if needed
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles all audio processing and transcription tasks with GPU optimizations"""
    
    def __init__(self, model_size: str = "base"):
        """Initialize with Whisper model and GPU detection"""
        self.model_size = model_size
        self.whisper_model = None
        self.model_lock = threading.Lock()  # Thread safety for model access
        self.device = self._detect_device()
        self.load_whisper_model()
    
    def _detect_device(self) -> str:
        """Detect and log the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU Detected: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"✅ Using CUDA acceleration")
        else:
            device = "cpu"
            logger.info("⚠️ No CUDA GPU detected, using CPU")
        
        return device
    
    def load_whisper_model(self):
        """Load Whisper model for transcription with GPU support"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            with self.model_lock:
                # Explicitly specify device for Whisper
                self.whisper_model = whisper.load_model(self.model_size, device=self.device)
            
            if self.device == "cuda":
                logger.info("🎯 Whisper model loaded on GPU - expect 5-10x faster transcription!")
            else:
                logger.info("📊 Whisper model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU if GPU fails
            if self.device == "cuda":
                logger.warning("GPU failed, falling back to CPU...")
                self.device = "cpu"
                self.whisper_model = whisper.load_model(self.model_size, device="cpu")
            else:
                raise Exception(f"Could not load Whisper model: {e}")
    
    def trim_audio_to_duration(self, audio_path: str, max_duration: float = 90.0) -> str:
        """Trim audio to fit within reel duration limits"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            max_samples = int(max_duration * sample_rate)
            
            if len(audio_data) > max_samples:
                logger.info(f"Trimming audio from {len(audio_data)/sample_rate:.2f}s to {max_duration}s")
                trimmed_audio = audio_data[:max_samples]
                trimmed_path = tempfile.mktemp(suffix='.wav')
                sf.write(trimmed_path, trimmed_audio, sample_rate)
                return trimmed_path
            else:
                logger.info("Audio duration is within limits, no trimming needed")
                return audio_path
                
        except Exception as e:
            logger.error(f"Audio trimming failed: {e}")
            return audio_path
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """Transcribe audio and extract word-level timestamps with GPU acceleration"""
        try:
            logger.info(f"🎤 Starting audio transcription on {self.device.upper()}...")
            
            # Show GPU memory usage if using CUDA
            if self.device == "cuda":
                gpu_memory_before = torch.cuda.memory_allocated(0) / 1024**2
                logger.info(f"GPU Memory before: {gpu_memory_before:.1f}MB")
            
            start_time = time.time()
            
            with self.model_lock:  # Thread-safe model access
                result = self.whisper_model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    verbose=False,
                    language="en"
                )
            
            transcription_time = time.time() - start_time
            
            # Show performance stats
            if self.device == "cuda":
                gpu_memory_after = torch.cuda.memory_allocated(0) / 1024**2
                logger.info(f"GPU Memory after: {gpu_memory_after:.1f}MB")
                logger.info(f"⚡ GPU Transcription completed in {transcription_time:.2f}s")
            else:
                logger.info(f"💻 CPU Transcription completed in {transcription_time:.2f}s")
            
            words_with_timing = []
            full_text = ""
            
            for segment in result.get('segments', []):
                if 'words' in segment and segment['words']:
                    for word_data in segment['words']:
                        word_info = {
                            'text': word_data.get('word', '').strip(),
                            'start': word_data.get('start', 0),
                            'end': word_data.get('end', 0),
                            'confidence': word_data.get('probability', 0.9)
                        }
                        words_with_timing.append(word_info)
                        full_text += word_info['text'] + " "
            
            total_duration = result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0
            
            transcription_result = {
                'text': full_text.strip(),
                'words': words_with_timing,
                'duration': total_duration,
                'language': result.get('language', 'en'),
                'segments_count': len(result.get('segments', [])),
                'device_used': self.device,
                'transcription_time': transcription_time
            }
            
            logger.info(f"✅ Transcription complete: {len(words_with_timing)} words, {total_duration:.2f}s duration")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'text': '',
                'words': [],
                'duration': 0,
                'language': 'en',
                'segments_count': 0,
                'device_used': self.device,
                'transcription_time': 0,
                'error': str(e)
            }
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate that audio file is readable and has content"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            if len(audio_data) == 0:
                logger.error("Audio file is empty")
                return False
            
            duration = len(audio_data) / sample_rate
            if duration < 1.0:
                logger.error(f"Audio file too short: {duration:.2f}s")
                return False
            
            logger.info(f"Audio file validated: {duration:.2f}s, {sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    def get_device_info(self) -> Dict:
        """Get current device information"""
        info = {"device": self.device}
        
        if self.device == "cuda" and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f}MB",
                "cuda_version": torch.version.cuda
            })
        
        return info