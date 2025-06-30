"""
Enhanced video processing module with NO FADE EFFECTS + REDDIT INTEGRATION
Text appears instantly and stays visible - no vanishing animations
Font rendering matches preview exactly
NEW: Reddit post image overlay during title phase
FIXED: Reddit image sizing, text margins, heartbeat speed
UPDATED: Added dead air support - last text remains visible during ending silence
FIXED: Video duration preserved for dead air (no cutting with FFmpeg)
FINAL: Fixed text disappearing during dead air - text now stays visible
FIXED: Dynamic font sizing and permanent margins for 1-5 words per segment
"""

import cv2
import numpy as np
import tempfile
import os
import logging
import subprocess
import threading
import concurrent.futures
import math
import time
import json  # For FFprobe output parsing
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import gc
import shutil

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages temporary files and prevents hard drive bloat"""
    
    def __init__(self):
        self.current_session_files = []
        self.lock = threading.Lock()
    
    def create_temp_file(self, suffix='.mp4'):
        """Create a temporary file and track it"""
        temp_file = tempfile.mktemp(suffix=suffix)
        with self.lock:
            self.current_session_files.append(temp_file)
        return temp_file
    
    def cleanup_session(self):
        """Clean up all files from current session"""
        with self.lock:
            for file_path in self.current_session_files:
                try:
                    if os.path.exists(file_path):
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            logger.info(f"Cleaned up temp dir: {file_path}")
                        else:
                            os.remove(file_path)
                            logger.info(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
            self.current_session_files.clear()
    
    def make_permanent(self, temp_path: str, permanent_path: str):
        """Move temp file to permanent location for download"""
        try:
            shutil.copy2(temp_path, permanent_path)
            logger.info(f"Saved permanent copy: {permanent_path}")
            return permanent_path
        except Exception as e:
            logger.error(f"Failed to save permanent copy: {e}")
            return temp_path

# Global cache manager instance
cache_manager = CacheManager()

def calculate_dynamic_font_size_for_video(base_font_size: int, words_per_segment: int, text_length: int = 0) -> int:
    """
    FIXED: Calculate dynamic font size for video rendering based on words per segment and text length
    
    Args:
        base_font_size: Original font size setting
        words_per_segment: Number of words per segment (1-5)
        text_length: Total character length of text for additional scaling
        
    Returns:
        Adjusted font size for video
    """
    # Base scaling factors for different word counts
    scaling_factors = {
        1: 1.0,    # 1 word - full size
        2: 0.95,   # 2 words - slightly smaller
        3: 0.85,   # 3 words - moderate reduction
        4: 0.75,   # 4 words - significant reduction
        5: 0.65    # 5 words - most reduction
    }
    
    base_scale = scaling_factors.get(words_per_segment, 0.65)
    
    # Additional scaling based on text length
    if text_length > 0:
        if text_length > 50:
            length_scale = 0.85  # Very long text
        elif text_length > 35:
            length_scale = 0.90  # Long text
        elif text_length > 25:
            length_scale = 0.95  # Medium text
        else:
            length_scale = 1.0   # Short text
    else:
        length_scale = 1.0
    
    # Combine scaling factors
    final_scale = base_scale * length_scale
    adjusted_size = int(base_font_size * final_scale)
    
    # Ensure minimum readable size
    adjusted_size = max(24, adjusted_size)  # Minimum 24px for video readability
    
    logger.debug(f"Dynamic font sizing: {base_font_size}px -> {adjusted_size}px (words: {words_per_segment}, text: {text_length}chars)")
    return adjusted_size

class ProVideoProcessor:
    """Enhanced professional video processor with NO FADE EFFECTS + Reddit integration + Dead air support + Dynamic font sizing"""
    
    def __init__(self):
        """Initialize video processor with professional settings"""
        self.target_resolution = (1080, 1920)  # 9:16 aspect ratio for reels
        self.target_fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.thread_pool_size = min(8, os.cpu_count() or 4)
        
        # Enhanced font options with fallbacks
        self.font_options = {
            'Arial': ['arial.ttf', 'Arial.ttf'],
            'Arial Bold': ['arialbd.ttf', 'Arial Bold.ttf'],
            'Impact': ['impact.ttf', 'Impact.ttf'],
            'Times New Roman': ['times.ttf', 'Times New Roman.ttf'],
            'Helvetica': ['helvetica.ttf', 'Helvetica.ttf', 'arial.ttf'],
            'Georgia': ['georgia.ttf', 'Georgia.ttf', 'arial.ttf'],
            'Trebuchet MS': ['trebuc.ttf', 'Trebuchet MS.ttf', 'arial.ttf'],
            'Verdana': ['verdana.ttf', 'Verdana.ttf', 'arial.ttf'],
        }
        
        # Animation settings for heartbeat effect - 90% slower (was 60%)
        self.heartbeat_frequency = 0.05  # Much slower heartbeat (0.19 Hz = 11.4 BPM)
        self.heartbeat_amplitude = 0.03  # Subtle 3% size variation - same as preview
        
        # FIXED: Margin settings - permanent 10% margins
        self.safe_margin_percentage = 0.10  # 10% margins on all sides
        
    def cleanup_previous_session(self):
        """Clean up previous generation before starting new one"""
        cache_manager.cleanup_session()
        gc.collect()
        
    def process_background_video(self, video_path: str, target_duration: float, vignette_strength: float = 0.0) -> str:
        """
        Process background video with optional vignette effect
        
        Args:
            video_path: Path to input video
            target_duration: EXACT duration needed (includes dead air)
            vignette_strength: 0.0 to 1.0 for vignette darkness
        """
        try:
            logger.info(f"Processing background video with vignette: {vignette_strength}")
            logger.info(f"🕐 Target duration: {target_duration:.2f}s (includes dead air)")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_duration = frame_count / original_fps if original_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Original video: {width}x{height}, {original_duration:.2f}s")
            logger.info(f"Will generate {int(target_duration * self.target_fps)} frames at {self.target_fps}fps")
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            out = cv2.VideoWriter(output_path, self.fourcc, self.target_fps, self.target_resolution)
            
            # Create vignette mask if needed
            vignette_mask = None
            if vignette_strength > 0:
                vignette_mask = self._create_vignette_mask(self.target_resolution, vignette_strength)
            
            self._process_video_frames_with_vignette(cap, out, target_duration, original_duration, original_fps, vignette_mask)
            
            cap.release()
            out.release()
            
            # Verify the output duration
            verify_cap = cv2.VideoCapture(output_path)
            if verify_cap.isOpened():
                verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                verify_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                verify_duration = verify_frame_count / verify_fps if verify_fps > 0 else 0
                verify_cap.release()
                
                logger.info(f"✅ Background video processed:")
                logger.info(f"  • Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
                logger.info(f"  • Target duration: {target_duration:.2f}s")
                logger.info(f"  • Actual duration: {verify_duration:.2f}s")
                logger.info(f"  • Frame count: {verify_frame_count}")
                
                if abs(verify_duration - target_duration) > 0.5:
                    logger.warning(f"Duration mismatch: expected {target_duration:.2f}s, got {verify_duration:.2f}s")
            else:
                logger.info(f"Processed video: {self.target_resolution[0]}x{self.target_resolution[1]}, {target_duration:.2f}s target")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise Exception(f"Could not process background video: {e}")
    
    def _create_vignette_mask(self, resolution: Tuple[int, int], strength: float) -> np.ndarray:
        """Create vignette mask for background dimming"""
        width, height = resolution
        center_x, center_y = width // 2, height // 2
        
        # Create coordinate grids
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance
        normalized_distance = distance / max_distance
        
        # Create vignette effect
        vignette = 1.0 - (normalized_distance * strength)
        vignette = np.clip(vignette, 0.1, 1.0)  # Don't go completely black
        
        # Convert to 3-channel
        vignette_mask = np.stack([vignette, vignette, vignette], axis=2)
        
        return vignette_mask.astype(np.float32)
    
    def _process_video_frames_with_vignette(self, cap: cv2.VideoCapture, out: cv2.VideoWriter, 
                                          target_duration: float, original_duration: float, original_fps: float, vignette_mask):
        """Process video frames with vignette effect"""
        target_frames = int(target_duration * self.target_fps)
        processed_frames = 0
        batch_size = 30
        frame_batch = []
        
        while processed_frames < target_frames:
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_batch.append(frame)
            
            if len(frame_batch) >= batch_size or processed_frames + len(frame_batch) >= target_frames:
                processed_batch = self._process_frame_batch_with_vignette(frame_batch, vignette_mask)
                
                for processed_frame in processed_batch:
                    if processed_frames < target_frames:
                        out.write(processed_frame)
                        processed_frames += 1
                
                frame_batch.clear()
            
            if original_duration > target_duration * 1.5:
                skip_frames = max(1, int(original_fps / self.target_fps)) - 1
                for _ in range(skip_frames):
                    cap.read()
    
    def _process_frame_batch_with_vignette(self, frames: List[np.ndarray], vignette_mask) -> List[np.ndarray]:
        """Process frame batch with vignette"""
        if len(frames) <= 4:
            return [self._resize_and_apply_vignette(frame, vignette_mask) for frame in frames]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
            processed_frames = list(executor.map(lambda f: self._resize_and_apply_vignette(f, vignette_mask), frames))
        
        return processed_frames
    
    def _resize_and_apply_vignette(self, frame: np.ndarray, vignette_mask) -> np.ndarray:
        """Resize frame and apply vignette"""
        # First resize to target resolution
        frame = self._resize_to_vertical_format(frame)
        
        # Apply vignette if provided
        if vignette_mask is not None:
            frame = frame.astype(np.float32) / 255.0
            frame = frame * vignette_mask
            frame = (frame * 255.0).astype(np.uint8)
        
        return frame
    
    def _resize_to_vertical_format(self, frame: np.ndarray) -> np.ndarray:
        """Resize video frame to vertical format with smart cropping"""
        height, width = frame.shape[:2]
        target_w, target_h = self.target_resolution
        target_ratio = target_h / target_w
        current_ratio = height / width
        
        if current_ratio < target_ratio:
            new_width = int(height / target_ratio)
            x_offset = (width - new_width) // 2
            frame = frame[:, x_offset:x_offset + new_width]
        elif current_ratio > target_ratio:
            new_height = int(width * target_ratio)
            y_offset = (height - new_height) // 2
            frame = frame[y_offset:y_offset + new_height, :]
        
        frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)
        return frame

    def compose_video_with_reddit(self, background_video_path: str, text_overlays: List[Dict], 
                                reddit_image: Image.Image, reddit_display_duration: float) -> str:
        """
        Compose video with Reddit post image during title phase + text overlays during main phase + dead air support
        
        Args:
            background_video_path: Path to processed background video
            text_overlays: List of text overlays for main phase (includes dead air extension)
            reddit_image: PIL Image of Reddit post
            reddit_display_duration: Duration to show Reddit image (title + delay)
            
        Returns:
            Path to composed video
        """
        try:
            logger.info(f"Composing video with Reddit integration - Reddit display: {reddit_display_duration:.2f}s")
            
            cap = cv2.VideoCapture(background_video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open background video: {background_video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            out = cv2.VideoWriter(output_path, self.fourcc, fps, (frame_width, frame_height))
            
            # Prepare Reddit image for overlay
            reddit_overlay = self._prepare_reddit_overlay(reddit_image, (frame_width, frame_height))
            
            frame_number = 0
            batch_size = 15
            frame_batch = []
            time_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_number / fps
                frame_batch.append(frame)
                time_batch.append(current_time)
                
                if len(frame_batch) >= batch_size:
                    processed_batch = self._process_reddit_integration_batch(
                        frame_batch, time_batch, text_overlays, reddit_overlay, reddit_display_duration
                    )
                    
                    for processed_frame in processed_batch:
                        out.write(processed_frame)
                    
                    frame_batch.clear()
                    time_batch.clear()
                
                frame_number += 1
            
            if frame_batch:
                processed_batch = self._process_reddit_integration_batch(
                    frame_batch, time_batch, text_overlays, reddit_overlay, reddit_display_duration
                )
                for processed_frame in processed_batch:
                    out.write(processed_frame)
            
            cap.release()
            out.release()
            
            logger.info(f"Reddit video composition complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Reddit video composition failed: {e}")
            return background_video_path

    def _prepare_reddit_overlay(self, reddit_image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Prepare Reddit image for video overlay - covers almost full screen
        
        Args:
            reddit_image: PIL Image of Reddit post
            target_size: Target video resolution (width, height)
            
        Returns:
            NumPy array ready for overlay
        """
        try:
            # Convert to RGB if needed
            if reddit_image.mode == 'RGBA':
                # Create white background for RGBA images
                white_bg = Image.new('RGB', reddit_image.size, (255, 255, 255))
                white_bg.paste(reddit_image, mask=reddit_image.split()[-1])  # Use alpha as mask
                reddit_image = white_bg
            elif reddit_image.mode != 'RGB':
                reddit_image = reddit_image.convert('RGB')
            
            # Calculate scaling to cover almost full screen with tiny margins
            video_width, video_height = target_size
            img_width, img_height = reddit_image.size
            
            # Scale to fit within 95% of video dimensions (tiny margins on sides)
            max_width = int(video_width * 0.95)
            max_height = int(video_height * 0.90)  # Slightly more margin on top/bottom
            
            # Calculate scale factor to fit within constraints
            width_scale = max_width / img_width
            height_scale = max_height / img_height
            scale_factor = min(width_scale, height_scale)
            
            # Resize Reddit image
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            reddit_resized = reddit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            reddit_array = np.array(reddit_resized)
            reddit_bgr = cv2.cvtColor(reddit_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Reddit overlay prepared: {new_width}x{new_height} (scale: {scale_factor:.2f}) - covers {(new_width/video_width)*100:.1f}% width")
            return reddit_bgr
            
        except Exception as e:
            logger.error(f"Failed to prepare Reddit overlay: {e}")
            # Return a simple colored rectangle as fallback
            fallback = np.zeros((int(target_size[1]*0.8), int(target_size[0]*0.9), 3), dtype=np.uint8)
            fallback[:] = (255, 69, 0)  # Reddit orange
            return fallback

    def _process_reddit_integration_batch(self, frames: List[np.ndarray], times: List[float], 
                                        text_overlays: List[Dict], reddit_overlay: np.ndarray, 
                                        reddit_display_duration: float) -> List[np.ndarray]:
        """Process frame batch with Reddit integration and dead air support"""
        if len(frames) <= 3:
            return [self._add_reddit_and_text_to_frame(frame, text_overlays, reddit_overlay, reddit_display_duration, time) 
                   for frame, time in zip(frames, times)]
        
        def process_single_frame(frame_time_tuple):
            frame, time = frame_time_tuple
            return self._add_reddit_and_text_to_frame(frame, text_overlays, reddit_overlay, reddit_display_duration, time)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, self.thread_pool_size)) as executor:
            processed_frames = list(executor.map(process_single_frame, zip(frames, times)))
        
        return processed_frames

    def _add_reddit_and_text_to_frame(self, frame: np.ndarray, text_overlays: List[Dict], 
                                     reddit_overlay: np.ndarray, reddit_display_duration: float, 
                                     current_time: float) -> np.ndarray:
        """
        UPDATED: Add Reddit image and/or text overlays to frame based on timing (with dead air support)
        
        Args:
            frame: Video frame
            text_overlays: List of text overlays (includes dead air extension for last segment)
            reddit_overlay: Prepared Reddit image
            reddit_display_duration: Duration to show Reddit image
            current_time: Current time in video
            
        Returns:
            Frame with overlays applied
        """
        # Phase 1: Reddit image display (0 to reddit_display_duration)
        if current_time <= reddit_display_duration:
            frame = self._overlay_reddit_image(frame, reddit_overlay)
        
        # Phase 2: Text overlays (after reddit_display_duration) - includes dead air
        if current_time > reddit_display_duration:
            active_texts = [
                overlay for overlay in text_overlays
                if overlay['start_time'] <= current_time <= overlay['end_time']
            ]
            
            if active_texts:
                # Convert to PIL for text rendering
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                for text_info in active_texts:
                    # INSTANT DISPLAY - no opacity calculation, only heartbeat scale
                    opacity = 1.0  # ALWAYS FULL OPACITY (even during dead air)
                    scale_factor = self._calculate_heartbeat_scale(current_time)
                    
                    pil_image = self._draw_text_instant_display_with_dynamic_sizing(pil_image, text_info, opacity, scale_factor)
                
                # Convert back to OpenCV format
                rgb_array = np.array(pil_image)
                frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return frame

    def _overlay_reddit_image(self, frame: np.ndarray, reddit_overlay: np.ndarray) -> np.ndarray:
        """
        Overlay Reddit image on video frame (centered)
        
        Args:
            frame: Video frame
            reddit_overlay: Prepared Reddit image
            
        Returns:
            Frame with Reddit image overlaid
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            overlay_height, overlay_width = reddit_overlay.shape[:2]
            
            # Calculate center position
            x_offset = (frame_width - overlay_width) // 2
            y_offset = (frame_height - overlay_height) // 2
            
            # Ensure we don't go out of bounds
            x_offset = max(0, min(x_offset, frame_width - overlay_width))
            y_offset = max(0, min(y_offset, frame_height - overlay_height))
            
            # Create a copy of the frame to avoid modifying original
            result_frame = frame.copy()
            
            # Overlay the Reddit image
            result_frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = reddit_overlay
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Failed to overlay Reddit image: {e}")
            return frame

    def add_prepared_audio_track(self, video_path: str, prepared_audio_path: str) -> str:
        """
        FIXED: Add pre-prepared audio track without cutting video (preserves dead air video)
        
        Args:
            video_path: Path to video file (longer than audio to include dead air)
            prepared_audio_path: Path to pre-prepared audio with all timing included
            
        Returns:
            Path to video with audio track (video length preserved for dead air)
        """
        try:
            logger.info("Adding pre-prepared audio track while preserving video length for dead air")
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            
            # FIXED: Remove -shortest flag to preserve video length for dead air
            # Use video length as master timeline, pad audio if needed
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', prepared_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',  # Use video stream from first input
                '-map', '1:a:0',  # Use audio stream from second input
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                logger.warning("Trying alternative approach...")
                
                # Alternative approach: pad audio to match video length
                return self._add_audio_with_padding(video_path, prepared_audio_path)
            
            logger.info("✅ Pre-prepared audio track added while preserving video length for dead air")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout - audio processing took too long")
            return video_path
        except Exception as e:
            logger.error(f"Failed to add pre-prepared audio track: {e}")
            return video_path
    
    def _add_audio_with_padding(self, video_path: str, audio_path: str) -> str:
        """
        Alternative method: Add audio with padding to match video length
        """
        try:
            logger.info("Using alternative audio addition method with padding")
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            
            # Get video duration first
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                video_duration = float(video_info['format']['duration'])
                
                # Add audio with filter to pad it to video duration
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-filter_complex', f'[1:a]apad=whole_dur={video_duration}[audio]',
                    '-map', '0:v',
                    '-map', '[audio]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-y',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    logger.info("✅ Audio added with padding to match video duration")
                    return output_path
            
            # Final fallback
            logger.warning("Audio padding failed, using video without audio track")
            return video_path
            
        except Exception as e:
            logger.error(f"Alternative audio addition failed: {e}")
            return video_path

    def add_dual_audio_tracks(self, video_path: str, title_audio_path: str, main_audio_path: str, 
                            reddit_display_duration: float) -> str:
        """
        Add dual audio tracks with proper timing (title + delay + main)
        
        Args:
            video_path: Path to video file
            title_audio_path: Path to title audio
            main_audio_path: Path to main audio  
            reddit_display_duration: Duration of Reddit display phase
            
        Returns:
            Path to video with audio tracks
        """
        try:
            logger.info("Adding dual audio tracks with proper timing")
            
            # Create temporary files for audio processing
            silence_path = cache_manager.create_temp_file(suffix='.wav')
            combined_audio_path = cache_manager.create_temp_file(suffix='.wav')
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            
            # Calculate delay duration (reddit_display_duration - title_duration)
            from audio_processor import AudioProcessor
            audio_proc = AudioProcessor()
            title_duration = audio_proc.get_audio_duration(title_audio_path)
            delay_duration = reddit_display_duration - title_duration
            
            # Create silence for delay
            if delay_duration > 0:
                silence_path = audio_proc.create_silence_audio(delay_duration)
                
                if silence_path is None:
                    logger.error("Failed to create silence audio")
                    return video_path
                
                # Combine: title + silence + main
                combine_cmd = [
                    'ffmpeg',
                    '-i', title_audio_path,
                    '-i', silence_path,
                    '-i', main_audio_path,
                    '-filter_complex', '[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]',
                    '-map', '[out]',
                    '-y',
                    combined_audio_path
                ]
            else:
                # No delay needed, just combine title + main
                combine_cmd = [
                    'ffmpeg',
                    '-i', title_audio_path,
                    '-i', main_audio_path,
                    '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
                    '-map', '[out]',
                    '-y',
                    combined_audio_path
                ]
            
            # Combine audio files
            result = subprocess.run(combine_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Audio combination failed: {result.stderr}")
                return video_path
            
            # Add combined audio to video
            final_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', combined_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-shortest',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-y',
                output_path
            ]
            
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Final audio addition failed: {result.stderr}")
                return video_path
            
            logger.info("✅ Dual audio tracks added successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add dual audio tracks: {e}")
            return video_path
    
    def add_audio_track(self, video_path: str, audio_path: str) -> str:
        """Add audio track with optimized FFmpeg settings"""
        try:
            logger.info("Adding audio track to video")
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-shortest',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                logger.warning("Continuing without audio track")
                return video_path
            
            logger.info("Audio track added successfully")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout - audio processing took too long")
            return video_path
        except Exception as e:
            logger.error(f"Failed to add audio track: {e}")
            return video_path
    
    def create_text_overlays(self, segments: List[Dict], style_config: Dict) -> List[Dict]:
        """FIXED: Create text overlay information with NO FADE EFFECTS, dynamic font sizing, and dead air support"""
        text_overlays = []
        
        logger.info(f"Creating {len(segments)} text overlays with INSTANT display, dynamic font sizing, and dead air support")
        
        # Extract words per segment for dynamic font sizing
        words_per_segment = style_config.get('words_per_segment', 2)
        
        for i, segment in enumerate(segments):
            try:
                text_overlay = self._create_text_overlay_with_dynamic_sizing(segment, style_config, i, len(segments), words_per_segment)
                if text_overlay:
                    text_overlays.append(text_overlay)
            except Exception as e:
                logger.error(f"Failed to create text overlay for segment {i}: {e}")
        
        text_overlays.sort(key=lambda x: x['start_time'])
        
        # Log dead air information for last segment
        if text_overlays and text_overlays[-1].get('has_dead_air_extension', False):
            last_overlay = text_overlays[-1]
            logger.info(f"🔥 DEAD AIR TEXT OVERLAY CONFIRMED with dynamic font sizing:")
            logger.info(f"  • Text: '{last_overlay['text'][:50]}{'...' if len(last_overlay['text']) > 50 else ''}'")
            logger.info(f"  • Start time: {last_overlay['start_time']:.2f}s")
            logger.info(f"  • End time: {last_overlay['end_time']:.2f}s")
            logger.info(f"  • Dynamic font size: {last_overlay['style']['dynamic_font_size']}px")
            logger.info(f"  • Dead air starts at: {last_overlay.get('dead_air_start', 'N/A'):.2f}s")
            logger.info(f"  • Dead air duration: {last_overlay.get('dead_air_duration', 'N/A'):.2f}s")
            logger.info(f"  • Text will remain visible during {last_overlay.get('dead_air_duration', 0):.2f}s dead air!")
        else:
            logger.warning("❌ No text overlay with dead air extension found!")
        
        logger.info(f"Successfully created {len(text_overlays)} text overlays with instant display, dynamic font sizing, and dead air support")
        return text_overlays
    
    def _create_text_overlay_with_dynamic_sizing(self, segment: Dict, style_config: Dict, segment_index: int, total_segments: int, words_per_segment: int) -> Optional[Dict]:
        """FIXED: Create text overlay with NO FADE EFFECTS, dynamic font sizing, and dead air support"""
        try:
            text = segment['text']
            start_time = segment['start']
            duration = segment['duration']
            
            # Apply text case transformation
            text_case = style_config.get('text_case', 'regular')
            if text_case == 'uppercase':
                text = text.upper()
            elif text_case == 'lowercase':
                text = text.lower()
            
            # FIXED: Calculate dynamic font size based on words per segment and text length
            base_font_size = style_config.get('font_size', 60)
            text_length = len(text)
            dynamic_font_size = calculate_dynamic_font_size_for_video(base_font_size, words_per_segment, text_length)
            
            # Smart text wrapping based on dynamic font size
            max_chars = self._calculate_max_chars_for_dynamic_font_size(dynamic_font_size, words_per_segment)
            if len(text) > max_chars:
                text = self._smart_wrap_text_for_margins(text, max_chars)
            
            # NO FADE EFFECTS - instant display
            display_end = start_time + duration
            
            text_overlay = {
                'text': text,
                'start_time': start_time,
                'duration': display_end - start_time,
                'end_time': display_end,
                'fade_in_duration': 0,  # NO FADE IN
                'fade_out_duration': 0,  # NO FADE OUT
                'style': {
                    'font_size': base_font_size,  # Keep original for reference
                    'dynamic_font_size': dynamic_font_size,  # NEW: Dynamic font size
                    'text_color': style_config.get('text_color', '#FFFFFF'),
                    'stroke_color': style_config.get('stroke_color', '#000000'),
                    'stroke_width': style_config.get('stroke_width', 3),
                    'font_family': style_config.get('font_family', 'Arial'),
                    'position': style_config.get('position', 'center'),
                    'text_case': style_config.get('text_case', 'regular'),
                    'words_per_segment': words_per_segment,  # For reference
                }
            }
            
            # UPDATED: Copy dead air information from segment
            if segment.get('has_dead_air_extension', False):
                text_overlay['has_dead_air_extension'] = True
                text_overlay['dead_air_start'] = segment.get('dead_air_start', 0)
                text_overlay['dead_air_duration'] = segment.get('dead_air_duration', 0)
                text_overlay['natural_end'] = segment.get('natural_end', 0)
            
            return text_overlay
            
        except Exception as e:
            logger.error(f"Failed to create text overlay info: {e}")
            return None
    
    def _calculate_max_chars_for_dynamic_font_size(self, dynamic_font_size: int, words_per_segment: int) -> int:
        """FIXED: Calculate maximum characters per line based on dynamic font size and word count"""
        # Base calculation: larger fonts need fewer characters, more words need better wrapping
        base_chars = 45
        
        # Font size factor (inverse relationship)
        size_factor = 60 / dynamic_font_size
        
        # Word count factor (more words = need more efficient wrapping)
        word_factor = {
            1: 1.2,   # 1 word - can be longer per line
            2: 1.1,   # 2 words - slightly longer
            3: 1.0,   # 3 words - normal
            4: 0.9,   # 4 words - shorter lines
            5: 0.8    # 5 words - much shorter lines
        }.get(words_per_segment, 0.8)
        
        max_chars = int(base_chars * size_factor * word_factor)
        
        # Ensure reasonable bounds
        max_chars = max(15, min(max_chars, 60))
        
        logger.debug(f"Max chars for font {dynamic_font_size}px, {words_per_segment} words: {max_chars}")
        return max_chars
    
    def _smart_wrap_text_for_margins(self, text: str, max_chars_per_line: int) -> str:
        """FIXED: Smart text wrapping that respects word boundaries and maintains margins"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Check if adding this word would exceed line length
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars_per_line:
                current_line.append(word)
            else:
                # If current line has words, finish it and start new line
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, break it carefully
                    if len(word) > max_chars_per_line:
                        # Break long word at reasonable point
                        break_point = max_chars_per_line - 1
                        lines.append(word[:break_point] + "-")
                        current_line = [word[break_point:]] if len(word) > break_point else []
                    else:
                        current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to maximum 3 lines to prevent overflow
        if len(lines) > 3:
            # Combine the last lines if we have too many
            combined_last = ' '.join(lines[2:])
            lines = lines[:2] + [combined_last[:max_chars_per_line * 2]]  # Limit combined line length
        
        return '\n'.join(lines)
    
    def compose_final_video(self, background_video_path: str, text_overlays: List[Dict]) -> str:
        """Compose final video with INSTANT text display, dynamic font sizing, and heartbeat animation"""
        try:
            logger.info("Composing final video with INSTANT text display, dynamic font sizing, and heartbeat animation")
            
            cap = cv2.VideoCapture(background_video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open background video: {background_video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            out = cv2.VideoWriter(output_path, self.fourcc, fps, (frame_width, frame_height))
            
            frame_number = 0
            batch_size = 15
            frame_batch = []
            time_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_number / fps
                frame_batch.append(frame)
                time_batch.append(current_time)
                
                if len(frame_batch) >= batch_size:
                    processed_batch = self._process_text_overlay_batch(frame_batch, time_batch, text_overlays)
                    
                    for processed_frame in processed_batch:
                        out.write(processed_frame)
                    
                    frame_batch.clear()
                    time_batch.clear()
                
                frame_number += 1
            
            if frame_batch:
                processed_batch = self._process_text_overlay_batch(frame_batch, time_batch, text_overlays)
                for processed_frame in processed_batch:
                    out.write(processed_frame)
            
            cap.release()
            out.release()
            
            logger.info(f"Video composition complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video composition failed: {e}")
            return background_video_path
    
    def _process_text_overlay_batch(self, frames: List[np.ndarray], times: List[float], text_overlays: List[Dict]) -> List[np.ndarray]:
        """Process text overlay batch with animations and dynamic font sizing"""
        if len(frames) <= 3:
            return [self._add_text_overlays_to_frame(frame, text_overlays, time) 
                   for frame, time in zip(frames, times)]
        
        def process_single_frame(frame_time_tuple):
            frame, time = frame_time_tuple
            return self._add_text_overlays_to_frame(frame, text_overlays, time)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, self.thread_pool_size)) as executor:
            processed_frames = list(executor.map(process_single_frame, zip(frames, times)))
        
        return processed_frames
    
    def _add_text_overlays_to_frame(self, frame: np.ndarray, text_overlays: List[Dict], current_time: float) -> np.ndarray:
        """FIXED: Add text overlays to frame with INSTANT display, dynamic font sizing, and heartbeat animation (dead air support)"""
        active_texts = [
            overlay for overlay in text_overlays
            if overlay['start_time'] <= current_time <= overlay['end_time']
        ]
        
        # DEBUG: Log dead air text visibility
        if current_time > 0 and any(overlay.get('has_dead_air_extension', False) for overlay in text_overlays):
            dead_air_overlay = next((o for o in text_overlays if o.get('has_dead_air_extension', False)), None)
            if dead_air_overlay:
                dead_air_start = dead_air_overlay.get('dead_air_start', 0)
                if current_time >= dead_air_start:
                    logger.debug(f"🔥 DEAD AIR ACTIVE at {current_time:.2f}s - Text: '{dead_air_overlay['text'][:30]}...' (ends at {dead_air_overlay['end_time']:.2f}s)")
        
        if not active_texts:
            return frame
        
        # Convert to PIL for text rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        for text_info in active_texts:
            # INSTANT DISPLAY - no opacity calculation, only heartbeat scale (works during dead air too)
            opacity = 1.0  # ALWAYS FULL OPACITY
            scale_factor = self._calculate_heartbeat_scale(current_time)
            
            pil_image = self._draw_text_instant_display_with_dynamic_sizing(pil_image, text_info, opacity, scale_factor)
        
        # Convert back to OpenCV format
        rgb_array = np.array(pil_image)
        bgr_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_frame
    
    def _calculate_heartbeat_scale(self, current_time: float) -> float:
        """Calculate subtle heartbeat scale effect - 90% slower (works during dead air)"""
        # Much slower heartbeat: sin wave with very low frequency
        heartbeat_phase = math.sin(2 * math.pi * self.heartbeat_frequency * current_time)
        scale_variation = 1.0 + (heartbeat_phase * self.heartbeat_amplitude)
        return scale_variation
    
    def _draw_text_instant_display_with_dynamic_sizing(self, pil_image: Image.Image, text_info: Dict, opacity: float, scale_factor: float) -> Image.Image:
        """FIXED: Draw text with INSTANT display, dynamic font sizing, and permanent margins (works during dead air)"""
        try:
            text = text_info['text']
            style = text_info['style']
            
            # FIXED: Use dynamic font size instead of base font size
            dynamic_font_size = style.get('dynamic_font_size', style.get('font_size', 60))
            
            # Get styling with heartbeat scale applied to dynamic font size
            font_size = int(dynamic_font_size * scale_factor)  # Apply heartbeat scale to dynamic size
            text_color = self._safe_hex_to_rgb(style['text_color'])
            stroke_color = self._safe_hex_to_rgb(style['stroke_color'])
            stroke_width = style['stroke_width']
            
            # Load font
            font = self._load_font(style['font_family'], font_size)
            
            # Create a separate image for text with transparency
            text_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_image)
            
            # Handle multi-line text
            lines = text.split('\n') if '\n' in text else [text]
            
            # Calculate text dimensions with PERMANENT margins
            line_heights = []
            line_widths = []
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_widths.append(bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])
            
            max_width = max(line_widths) if line_widths else 0
            total_height = sum(line_heights) + (len(lines) - 1) * 8  # 8px line spacing
            
            # FIXED: Calculate safe positioning with PERMANENT margins (10% on each side)
            image_width, image_height = pil_image.size
            
            # FIXED: PERMANENT 10% margins - never changes regardless of word count
            safe_margin_x = int(image_width * self.safe_margin_percentage)
            safe_margin_y = int(image_height * self.safe_margin_percentage)
            safe_width = image_width - (2 * safe_margin_x)
            safe_height = image_height - (2 * safe_margin_y)
            
            # Get base position within safe area
            position_type = style['position']
            
            if position_type == 'center':
                base_x = safe_margin_x + (safe_width - max_width) // 2
                base_y = safe_margin_y + (safe_height - total_height) // 2
            elif position_type == 'bottom':
                base_x = safe_margin_x + (safe_width - max_width) // 2
                base_y = safe_margin_y + int(safe_height * 0.75) - total_height
            elif position_type == 'top':
                base_x = safe_margin_x + (safe_width - max_width) // 2
                base_y = safe_margin_y + int(safe_height * 0.15)
            else:
                base_x = safe_margin_x + (safe_width - max_width) // 2
                base_y = safe_margin_y + (safe_height - total_height) // 2
            
            # FIXED: Ensure text NEVER goes beyond permanent safe bounds
            base_x = max(safe_margin_x, min(base_x, safe_margin_x + safe_width - max_width))
            base_y = max(safe_margin_y, min(base_y, safe_margin_y + safe_height - total_height))
            
            # Additional safety check - if text is still too wide, truncate or wrap further
            if max_width > safe_width:
                logger.warning(f"Text width {max_width}px exceeds safe width {safe_width}px - adjusting font size")
                # Reduce font size further if text still doesn't fit
                reduction_factor = safe_width / max_width * 0.9  # 90% of safe width
                font_size = int(font_size * reduction_factor)
                font = self._load_font(style['font_family'], font_size)
                
                # Recalculate with smaller font
                line_widths = []
                line_heights = []
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_widths.append(bbox[2] - bbox[0])
                    line_heights.append(bbox[3] - bbox[1])
                
                max_width = max(line_widths) if line_widths else 0
                total_height = sum(line_heights) + (len(lines) - 1) * 8
                
                # Recalculate position
                base_x = safe_margin_x + (safe_width - max_width) // 2
                base_y = safe_margin_y + (safe_height - total_height) // 2
            
            # Draw each line with PERMANENT margins
            current_y = base_y
            for i, line in enumerate(lines):
                line_width = line_widths[i]
                # Center each line within safe area
                line_x = safe_margin_x + (safe_width - line_width) // 2
                
                # FIXED: Final safety check - ensure line never exceeds safe bounds
                line_x = max(safe_margin_x, min(line_x, safe_margin_x + safe_width - line_width))
                
                # Draw stroke with dynamic scaling
                stroke_with_alpha = (*stroke_color, 255)
                dynamic_stroke_width = max(1, int(stroke_width * (font_size / dynamic_font_size)))
                
                if dynamic_stroke_width > 0:
                    for dx in range(-dynamic_stroke_width, dynamic_stroke_width + 1):
                        for dy in range(-dynamic_stroke_width, dynamic_stroke_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((line_x + dx, current_y + dy), line, 
                                        font=font, fill=stroke_with_alpha)
                
                # Draw main text
                text_with_alpha = (*text_color, 255)
                draw.text((line_x, current_y), line, font=font, fill=text_with_alpha)
                
                current_y += line_heights[i] + 8
            
            # Composite text onto main image
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), text_image)
            return pil_image.convert('RGB')
                
        except Exception as e:
            logger.error(f"Failed to draw text with dynamic sizing: {e}")
            return pil_image
    
    def _safe_hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Safely convert hex color to RGB tuple with error handling"""
        try:
            # Clean up the hex color string
            hex_color = str(hex_color).strip()
            if not hex_color.startswith('#'):
                hex_color = '#' + hex_color
            
            # Remove any extra characters
            hex_color = hex_color[:7]  # Keep only #RRGGBB
            
            # Ensure we have a valid 6-character hex
            if len(hex_color) != 7:
                logger.warning(f"Invalid hex color length: {hex_color}, using white")
                return (255, 255, 255)
            
            # Parse the hex
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                logger.warning(f"Invalid hex color after cleanup: {hex_color}, using white")
                return (255, 255, 255)
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16) 
            b = int(hex_color[4:6], 16)
            
            return (r, g, b)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse hex color '{hex_color}': {e}, using white")
            return (255, 255, 255)  # Default to white
    
    def _load_font(self, font_family: str, font_size: int):
        """Load font with comprehensive fallbacks - SAME AS PREVIEW"""
        font_paths = self.font_options.get(font_family, ['arial.ttf'])
        
        # Add system-specific paths for each font
        all_paths = []
        for font_file in font_paths:
            all_paths.extend([
                font_file,  # Direct path
                f"C:/Windows/Fonts/{font_file}",  # Windows
                f"/System/Library/Fonts/{font_file}",  # Mac
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
                "arial.ttf",  # Ultimate fallback
            ])
        
        for path in all_paths:
            try:
                return ImageFont.truetype(path, font_size)
            except:
                continue
        
        # Ultimate fallback
        return ImageFont.load_default()
    
    def export_video(self, video_path: str, output_path: str = None) -> str:
        """Export video with professional quality settings"""
        try:
            if output_path is None:
                output_path = cache_manager.create_temp_file(suffix='.mp4')
            
            logger.info(f"Exporting video to: {output_path}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '20',  # Higher quality
                '-preset', 'fast',
                '-tune', 'film',
                '-movflags', '+faststart',
                '-r', str(self.target_fps),
                '-threads', str(self.thread_pool_size),
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg export error: {result.stderr}")
                import shutil
                shutil.copy2(video_path, output_path)
            
            logger.info("Video export completed successfully")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("Export timeout - video processing took too long")
            raise Exception("Video export timed out")
        except Exception as e:
            logger.error(f"Video export failed: {e}")
            raise Exception(f"Could not export video: {e}")
    
    def save_for_download(self, temp_video_path: str) -> str:
        """Save video for permanent download"""
        permanent_path = tempfile.mktemp(suffix='_download.mp4')
        return cache_manager.make_permanent(temp_video_path, permanent_path)

# Create aliases for backward compatibility
VideoProcessor = ProVideoProcessor
EnhancedVideoProcessor = ProVideoProcessor