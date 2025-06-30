"""
Audio processing module for Whisper AI transcription
WITH GPU ACCELERATION and optimizations for faster processing
Enhanced with dual audio support for Reddit integration
FIXED VERSION - All errors resolved
UPDATED: Enhanced dead air support with 3.5-second ending silence
FIXED: Proper audio composition with exact timing for video synchronization
"""

import whisper
import soundfile as sf
import tempfile
import logging
import threading
import torch
import warnings
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# Suppress specific Triton warnings if needed
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles all audio processing and transcription tasks with GPU optimizations + dual audio support + dead air"""
    
    def __init__(self, model_size: str = "base"):
        """Initialize with Whisper model and GPU detection"""
        self.model_size = model_size
        self.whisper_model = None
        self.model_lock = threading.Lock()  # Thread safety for model access
        self.device = self._detect_device()
        self.load_whisper_model()
        
        # Dead air configuration - EXACTLY 3.5 seconds
        self.default_ending_silence = 3.5  # Default 3.5 seconds of dead air
    
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
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file without full processing"""
        try:
            audio_data, sample_rate = sf.read(audio_path)
            duration = len(audio_data) / sample_rate
            logger.info(f"Audio duration: {duration:.2f}s ({audio_path})")
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
    
    def validate_dual_audio_files(self, title_audio_path: str, main_audio_path: str) -> Tuple[bool, str]:
        """Validate both audio files for dual audio system"""
        try:
            # Validate title audio
            if not self.validate_audio_file(title_audio_path):
                return False, "❌ Invalid title audio file"
            
            # Validate main audio
            if not self.validate_audio_file(main_audio_path):
                return False, "❌ Invalid main audio file"
            
            # Get durations
            title_duration = self.get_audio_duration(title_audio_path)
            main_duration = self.get_audio_duration(main_audio_path)
            
            # Check reasonable durations
            if title_duration < 0.5:
                return False, "❌ Title audio too short (minimum 0.5s)"
            
            if main_duration < 1.0:
                return False, "❌ Main audio too short (minimum 1s)"
            
            if title_duration > 30.0:
                return False, "❌ Title audio too long (maximum 30s)"
            
            if main_duration > 300.0:
                return False, "❌ Main audio too long (maximum 5 minutes)"
            
            logger.info(f"✅ Dual audio validation passed - Title: {title_duration:.2f}s, Main: {main_duration:.2f}s")
            return True, f"✅ Audio files validated - Title: {title_duration:.2f}s, Main: {main_duration:.2f}s"
            
        except Exception as e:
            logger.error(f"Dual audio validation failed: {e}")
            return False, f"❌ Validation error: {str(e)}"
    
    def process_dual_audio_system(self, title_audio_path: str, main_audio_path: str, ending_silence: float = None) -> Dict:
        """
        FIXED: Process both audio files for Reddit integration system with EXACT timing calculations
        
        Args:
            title_audio_path: Path to title audio
            main_audio_path: Path to main audio  
            ending_silence: Duration of ending silence for dead air (default 3.5s)
            
        Returns:
            Dictionary with EXACT timing information including dead air
        """
        try:
            # Use default ending silence if not specified
            if ending_silence is None:
                ending_silence = self.default_ending_silence
            
            logger.info(f"🎵 Processing dual audio system (title + main + {ending_silence:.1f}s dead air)")
            
            # Validate both files first
            is_valid, validation_message = self.validate_dual_audio_files(title_audio_path, main_audio_path)
            if not is_valid:
                return {
                    'success': False,
                    'error': validation_message,
                    'title_duration': 0.0,
                    'main_duration': 0.0,
                    'total_duration': 0.0,
                    'ending_silence': ending_silence
                }
            
            # Get EXACT durations
            title_duration = self.get_audio_duration(title_audio_path)
            
            # Transcribe main audio only
            main_transcription = self.transcribe_with_timestamps(main_audio_path)
            
            if not main_transcription['words']:
                return {
                    'success': False,
                    'error': '❌ Failed to transcribe main audio',
                    'title_duration': title_duration,
                    'main_duration': 0.0,
                    'total_duration': title_duration + 1.0 + ending_silence,
                    'ending_silence': ending_silence
                }
            
            main_duration = main_transcription['duration']
            delay_duration = 1.0  # EXACTLY 1 second delay
            reddit_display_duration = title_duration + delay_duration
            
            # FIXED: Calculate EXACT durations for video synchronization
            audio_content_duration = reddit_display_duration + main_duration
            total_duration = audio_content_duration + ending_silence
            
            result = {
                'success': True,
                'title_duration': title_duration,
                'main_duration': main_duration,
                'delay_duration': delay_duration,
                'reddit_display_duration': reddit_display_duration,
                'ending_silence': ending_silence,
                'audio_content_duration': audio_content_duration,  # Duration without dead air
                'total_duration': total_duration,  # Duration including dead air
                'main_transcription': main_transcription,
                'title_audio_path': title_audio_path,
                'main_audio_path': main_audio_path,
                'dead_air_start': audio_content_duration,  # When dead air begins
                'phases': {
                    'title_phase': {'start': 0.0, 'end': title_duration},
                    'delay_phase': {'start': title_duration, 'end': reddit_display_duration},
                    'main_phase': {'start': reddit_display_duration, 'end': audio_content_duration},
                    'dead_air_phase': {'start': audio_content_duration, 'end': total_duration}
                }
            }
            
            logger.info(f"✅ Dual audio processing complete with EXACT timing:")
            logger.info(f"  • Title: {title_duration:.2f}s")
            logger.info(f"  • Delay: {delay_duration:.2f}s") 
            logger.info(f"  • Main: {main_duration:.2f}s")
            logger.info(f"  • Audio content ends: {audio_content_duration:.2f}s")
            logger.info(f"  • Dead air: {ending_silence:.2f}s")
            logger.info(f"  • Total duration: {total_duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Dual audio processing failed: {e}")
            return {
                'success': False,
                'error': f'❌ Processing error: {str(e)}',
                'title_duration': 0.0,
                'main_duration': 0.0,
                'total_duration': 0.0,
                'ending_silence': ending_silence or self.default_ending_silence
            }
    
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
            if duration < 0.1:  # Allow very short audio for title
                logger.error(f"Audio file too short: {duration:.2f}s")
                return False
            
            logger.info(f"Audio file validated: {duration:.2f}s, {sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    def create_silence_audio(self, duration: float, sample_rate: int = 44100) -> str:
        """Create EXACT silence audio file for delays and dead air"""
        try:
            # FIXED: Ensure EXACT duration by calculating exact samples
            silence_samples = int(duration * sample_rate)
            silence_data = np.zeros(silence_samples, dtype=np.float32)
            
            silence_path = tempfile.mktemp(suffix='.wav')
            sf.write(silence_path, silence_data, sample_rate)
            
            # Verify created duration
            created_duration = silence_samples / sample_rate
            logger.info(f"Created EXACT silence audio: {created_duration:.3f}s (requested: {duration:.3f}s)")
            return silence_path
            
        except Exception as e:
            logger.error(f"Failed to create silence audio: {e}")
            return None
    
    def combine_audio_files(self, audio_files: List[str], output_path: str = None) -> str:
        """Combine multiple audio files sequentially with EXACT timing"""
        try:
            if not audio_files:
                raise Exception("No audio files provided")
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Read all audio files
            combined_data = []
            sample_rate = None
            cumulative_duration = 0.0
            
            for i, audio_file in enumerate(audio_files):
                if audio_file is None:
                    continue
                    
                audio_data, sr = sf.read(audio_file)
                file_duration = len(audio_data) / sr
                
                # Ensure consistent sample rate
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}, resampling may be needed")
                
                combined_data.append(audio_data)
                cumulative_duration += file_duration
                
                logger.info(f"  • File {i+1}: {file_duration:.3f}s (cumulative: {cumulative_duration:.3f}s)")
            
            if not combined_data:
                raise Exception("No valid audio data found")
            
            # Concatenate all audio data
            final_audio = np.concatenate(combined_data)
            
            # Write combined audio
            sf.write(output_path, final_audio, sample_rate)
            
            # Verify final duration
            final_duration = len(final_audio) / sample_rate
            logger.info(f"Combined {len(audio_files)} audio files: {final_duration:.3f}s total")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to combine audio files: {e}")
            return audio_files[0] if audio_files else None
    
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
    
    def prepare_dual_audio_for_video(self, title_audio_path: str, main_audio_path: str, 
                                   delay_duration: float = 1.0, ending_silence: float = None) -> str:
        """
        FIXED: Prepare combined audio file for video with EXACT timing: title + delay + main + ending silence
        
        Args:
            title_audio_path: Path to title audio
            main_audio_path: Path to main audio
            delay_duration: Delay between title and main (default 1.0s)
            ending_silence: Silence at end for dead air (default 3.5s)
            
        Returns:
            Path to combined audio file with dead air
        """
        try:
            # Use default ending silence if not specified
            if ending_silence is None:
                ending_silence = self.default_ending_silence
            
            logger.info(f"Preparing dual audio for video with EXACT timing + {ending_silence:.1f}s dead air")
            
            # Get exact durations for logging
            title_duration = self.get_audio_duration(title_audio_path)
            main_duration = self.get_audio_duration(main_audio_path)
            
            # Create EXACT silence files
            delay_silence_path = self.create_silence_audio(delay_duration)
            ending_silence_path = self.create_silence_audio(ending_silence)
            
            if delay_silence_path is None or ending_silence_path is None:
                logger.warning("Failed to create silence, using main audio only")
                return main_audio_path
            
            # Combine with EXACT timing: title + delay_silence + main + ending_silence
            audio_files = [title_audio_path, delay_silence_path, main_audio_path, ending_silence_path]
            combined_path = self.combine_audio_files(audio_files)
            
            if combined_path:
                # Verify final timing
                total_duration = self.get_audio_duration(combined_path)
                expected_duration = title_duration + delay_duration + main_duration + ending_silence
                
                logger.info(f"✅ Dual audio prepared for video with EXACT timing:")
                logger.info(f"  • Title audio: {title_duration:.3f}s")
                logger.info(f"  • Delay: {delay_duration:.3f}s")
                logger.info(f"  • Main audio: {main_duration:.3f}s")
                logger.info(f"  • Dead air: {ending_silence:.3f}s")
                logger.info(f"  • Expected total: {expected_duration:.3f}s")
                logger.info(f"  • Actual total: {total_duration:.3f}s")
                
                if abs(total_duration - expected_duration) > 0.1:
                    logger.warning(f"Duration mismatch: expected {expected_duration:.3f}s, got {total_duration:.3f}s")
                
                return combined_path
            else:
                logger.warning("Failed to combine audio, using main audio only")
                return main_audio_path
                
        except Exception as e:
            logger.error(f"Failed to prepare dual audio: {e}")
            return main_audio_path
    
    def create_dead_air_audio(self, base_audio_path: str, dead_air_duration: float = None) -> str:
        """
        Create audio with dead air appended at the end
        
        Args:
            base_audio_path: Path to base audio file
            dead_air_duration: Duration of dead air to append (default 3.5s)
            
        Returns:
            Path to audio file with dead air appended
        """
        try:
            # Use default dead air duration if not specified
            if dead_air_duration is None:
                dead_air_duration = self.default_ending_silence
            
            logger.info(f"Creating audio with {dead_air_duration:.1f}s dead air")
            
            # Create dead air silence
            dead_air_path = self.create_silence_audio(dead_air_duration)
            
            if dead_air_path is None:
                logger.warning("Failed to create dead air, using original audio")
                return base_audio_path
            
            # Combine base audio + dead air
            audio_files = [base_audio_path, dead_air_path]
            combined_path = self.combine_audio_files(audio_files)
            
            if combined_path:
                base_duration = self.get_audio_duration(base_audio_path)
                total_duration = self.get_audio_duration(combined_path)
                
                logger.info(f"✅ Audio with dead air created:")
                logger.info(f"  • Original duration: {base_duration:.2f}s")
                logger.info(f"  • Dead air: {dead_air_duration:.2f}s")
                logger.info(f"  • Total duration: {total_duration:.2f}s")
                
                return combined_path
            else:
                logger.warning("Failed to combine audio with dead air")
                return base_audio_path
                
        except Exception as e:
            logger.error(f"Failed to create dead air audio: {e}")
            return base_audio_path
    
    def analyze_audio_composition(self, title_audio_path: str, main_audio_path: str, 
                                ending_silence: float = None) -> Dict:
        """
        Analyze the composition of a dual audio system including dead air
        
        Args:
            title_audio_path: Path to title audio
            main_audio_path: Path to main audio
            ending_silence: Duration of ending silence (default 3.5s)
            
        Returns:
            Dictionary with detailed composition analysis
        """
        try:
            # Use default ending silence if not specified
            if ending_silence is None:
                ending_silence = self.default_ending_silence
            
            title_duration = self.get_audio_duration(title_audio_path)
            main_duration = self.get_audio_duration(main_audio_path)
            delay_duration = 1.0
            
            # Calculate phase durations
            title_phase_duration = title_duration
            delay_phase_duration = delay_duration
            main_phase_duration = main_duration
            dead_air_duration = ending_silence
            
            # Calculate cumulative timings
            title_end = title_phase_duration
            delay_end = title_end + delay_phase_duration
            main_end = delay_end + main_phase_duration
            total_end = main_end + dead_air_duration
            
            composition = {
                'phases': {
                    'title': {
                        'duration': title_phase_duration,
                        'start': 0.0,
                        'end': title_end,
                        'description': 'Title audio plays, Reddit post displayed'
                    },
                    'delay': {
                        'duration': delay_phase_duration,
                        'start': title_end,
                        'end': delay_end,
                        'description': 'Silence delay, Reddit post still displayed'
                    },
                    'main': {
                        'duration': main_phase_duration,
                        'start': delay_end,
                        'end': main_end,
                        'description': 'Main audio plays, synchronized text displayed'
                    },
                    'dead_air': {
                        'duration': dead_air_duration,
                        'start': main_end,
                        'end': total_end,
                        'description': 'Dead air - last text remains visible, background continues'
                    }
                },
                'totals': {
                    'audio_content_duration': main_end,  # Duration of actual audio content
                    'total_duration': total_end,  # Duration including dead air
                    'dead_air_percentage': (dead_air_duration / total_end) * 100,
                    'active_audio_percentage': (main_end / total_end) * 100
                },
                'reddit_integration': {
                    'reddit_display_duration': delay_end,  # How long Reddit post is shown
                    'text_display_duration': main_phase_duration + dead_air_duration,  # How long text is shown
                    'text_start_time': delay_end  # When text starts appearing
                }
            }
            
            logger.info(f"📊 Audio composition analysis:")
            logger.info(f"  • Total duration: {total_end:.2f}s")
            logger.info(f"  • Audio content: {main_end:.2f}s ({composition['totals']['active_audio_percentage']:.1f}%)")
            logger.info(f"  • Dead air: {dead_air_duration:.2f}s ({composition['totals']['dead_air_percentage']:.1f}%)")
            
            return composition
            
        except Exception as e:
            logger.error(f"Failed to analyze audio composition: {e}")
            return {}
    
    def get_dead_air_info(self) -> Dict:
        """
        Get information about dead air configuration
        
        Returns:
            Dictionary with dead air configuration details
        """
        return {
            'default_duration': self.default_ending_silence,
            'purpose': 'Reflection time for viewers - last text remains visible',
            'behavior': 'Background video continues, text stays visible, audio is silent',
            'recommended_range': '2.0 - 5.0 seconds',
            'current_setting': f'{self.default_ending_silence:.1f} seconds'
        }
    
    def verify_dead_air_timing(self, prepared_audio_path: str, expected_dead_air_duration: float) -> bool:
        """
        NEW: Verify that prepared audio has correct dead air timing
        
        Args:
            prepared_audio_path: Path to prepared audio file
            expected_dead_air_duration: Expected duration of dead air
            
        Returns:
            True if timing is correct
        """
        try:
            # Read the audio file
            audio_data, sample_rate = sf.read(prepared_audio_path)
            total_duration = len(audio_data) / sample_rate
            
            # Check for silence at the end
            dead_air_samples = int(expected_dead_air_duration * sample_rate)
            
            if len(audio_data) < dead_air_samples:
                logger.warning("Audio file too short to contain expected dead air")
                return False
            
            # Check if the last portion is silent (within tolerance)
            end_portion = audio_data[-dead_air_samples:]
            max_amplitude = np.max(np.abs(end_portion))
            
            # Should be very quiet (near silence)
            if max_amplitude > 0.01:  # Tolerance for near-silence
                logger.warning(f"End portion not silent enough: max amplitude {max_amplitude:.4f}")
                return False
            
            logger.info(f"✅ Dead air timing verified: {expected_dead_air_duration:.2f}s of silence at end")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify dead air timing: {e}")
            return False