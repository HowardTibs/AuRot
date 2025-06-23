"""
Simplified video processing module
Clean and reliable text rendering without complex features
"""

import cv2
import numpy as np
import tempfile
import os
import logging
import subprocess
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
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

class ProVideoProcessor:
    """Simplified professional video processor"""
    
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
        
    def cleanup_previous_session(self):
        """Clean up previous generation before starting new one"""
        cache_manager.cleanup_session()
        gc.collect()
        
    def process_background_video(self, video_path: str, target_duration: float, vignette_strength: float = 0.0) -> str:
        """
        Process background video with optional vignette effect
        
        Args:
            video_path: Path to input video
            target_duration: Audio duration + 4 seconds
            vignette_strength: 0.0 to 1.0 for vignette darkness
        """
        try:
            logger.info(f"Processing background video with vignette: {vignette_strength}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_duration = frame_count / original_fps if original_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Original video: {width}x{height}, {original_duration:.2f}s")
            
            output_path = cache_manager.create_temp_file(suffix='.mp4')
            out = cv2.VideoWriter(output_path, self.fourcc, self.target_fps, self.target_resolution)
            
            # Create vignette mask if needed
            vignette_mask = None
            if vignette_strength > 0:
                vignette_mask = self._create_vignette_mask(self.target_resolution, vignette_strength)
            
            self._process_video_frames_with_vignette(cap, out, target_duration, original_duration, original_fps, vignette_mask)
            
            cap.release()
            out.release()
            
            logger.info(f"Processed video with vignette: {self.target_resolution[0]}x{self.target_resolution[1]}, {target_duration:.2f}s")
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
        """Create text overlay information"""
        text_overlays = []
        
        logger.info(f"Creating {len(segments)} text overlays")
        
        for i, segment in enumerate(segments):
            try:
                text_overlay = self._create_text_overlay(segment, style_config)
                if text_overlay:
                    text_overlays.append(text_overlay)
            except Exception as e:
                logger.error(f"Failed to create text overlay for segment {i}: {e}")
        
        text_overlays.sort(key=lambda x: x['start_time'])
        
        logger.info(f"Successfully created {len(text_overlays)} text overlays")
        return text_overlays
    
    def _create_text_overlay(self, segment: Dict, style_config: Dict) -> Optional[Dict]:
        """Create text overlay with styling info"""
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
            
            # Smart text wrapping based on font size
            font_size = style_config.get('font_size', 60)
            max_chars = self._calculate_max_chars_for_font_size(font_size)
            if len(text) > max_chars:
                text = self._smart_wrap_text(text, max_chars)
            
            text_overlay = {
                'text': text,
                'start_time': start_time,
                'duration': duration,
                'end_time': start_time + duration,
                'style': {
                    'font_size': font_size,
                    'text_color': style_config.get('text_color', '#FFFFFF'),
                    'stroke_color': style_config.get('stroke_color', '#000000'),
                    'stroke_width': style_config.get('stroke_width', 3),
                    'font_family': style_config.get('font_family', 'Arial'),
                    'position': style_config.get('position', 'center'),
                    'text_case': style_config.get('text_case', 'regular'),
                    'line_spacing': style_config.get('line_spacing', 10),
                    'letter_spacing': style_config.get('letter_spacing', 0),
                }
            }
            
            return text_overlay
            
        except Exception as e:
            logger.error(f"Failed to create text overlay info: {e}")
            return None
    
    def _calculate_max_chars_for_font_size(self, font_size: int) -> int:
        """Calculate maximum characters per line based on font size"""
        # Base calculation: larger fonts need fewer characters
        base_chars = 40
        size_factor = 60 / font_size  # Inverse relationship
        max_chars = int(base_chars * size_factor)
        return max(15, min(max_chars, 50))  # Keep between 15-50 chars
    
    def _smart_wrap_text(self, text: str, max_chars_per_line: int) -> str:
        """Smart text wrapping that respects word boundaries"""
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
                    # Single word is too long, break it
                    lines.append(word[:max_chars_per_line])
                    current_line = [word[max_chars_per_line:]] if len(word) > max_chars_per_line else []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def compose_final_video(self, background_video_path: str, text_overlays: List[Dict]) -> str:
        """Compose final video with clean text overlays"""
        try:
            logger.info("Composing final video with text overlays")
            
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
        """Process text overlay batch"""
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
        """Add text overlays to frame"""
        active_texts = [
            overlay for overlay in text_overlays
            if overlay['start_time'] <= current_time <= overlay['end_time']
        ]
        
        if not active_texts:
            return frame
        
        # Convert to PIL for text rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)
        
        for text_info in active_texts:
            self._draw_clean_text(draw, text_info, pil_image.size)
        
        # Convert back to OpenCV format
        rgb_array = np.array(pil_image)
        bgr_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_frame
    
    def _draw_clean_text(self, draw: ImageDraw.Draw, text_info: Dict, image_size: Tuple[int, int]):
        """Draw clean, professional text"""
        try:
            text = text_info['text']
            style = text_info['style']
            
            # Get styling
            font_size = style['font_size']
            text_color = self._safe_hex_to_rgb(style['text_color'])
            stroke_color = self._safe_hex_to_rgb(style['stroke_color'])
            stroke_width = style['stroke_width']
            line_spacing = style.get('line_spacing', 10)
            letter_spacing = style.get('letter_spacing', 0)
            
            # Load font
            font = self._load_font(style['font_family'], font_size)
            
            # Handle multi-line text with custom spacing and letter spacing
            lines = text.split('\n') if '\n' in text else [text]
            
            # Apply letter spacing if needed
            if letter_spacing > 0:
                lines = [self._apply_letter_spacing(line, letter_spacing) for line in lines]
            
            # Calculate text dimensions and position
            line_heights = []
            line_widths = []
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_widths.append(bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])
            
            max_width = max(line_widths) if line_widths else 0
            total_height = sum(line_heights) + (len(lines) - 1) * line_spacing
            
            # Get base position
            position_type = style['position']
            if position_type == 'center':
                base_x = (image_size[0] - max_width) // 2
                base_y = (image_size[1] - total_height) // 2
            elif position_type == 'bottom':
                base_x = (image_size[0] - max_width) // 2
                base_y = int(image_size[1] * 0.8) - total_height
            elif position_type == 'top':
                base_x = (image_size[0] - max_width) // 2
                base_y = int(image_size[1] * 0.15)
            else:
                base_x = (image_size[0] - max_width) // 2
                base_y = (image_size[1] - total_height) // 2
            
            # Draw each line
            current_y = base_y
            for i, line in enumerate(lines):
                line_width = line_widths[i]
                line_x = (image_size[0] - line_width) // 2
                
                # Draw stroke
                if stroke_width > 0:
                    for dx in range(-stroke_width, stroke_width + 1):
                        for dy in range(-stroke_width, stroke_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((line_x + dx, current_y + dy), line, font=font, fill=stroke_color)
                
                # Draw main text
                draw.text((line_x, current_y), line, font=font, fill=text_color)
                current_y += line_heights[i] + line_spacing
                
        except Exception as e:
            logger.error(f"Failed to draw text: {e}")
    
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
    
    def _apply_letter_spacing(self, text: str, spacing: int) -> str:
        """Apply letter spacing to text"""
        if spacing <= 0:
            return text
        
        # Add spaces between characters
        spaced_text = ""
        for i, char in enumerate(text):
            spaced_text += char
            if i < len(text) - 1 and char != ' ':
                spaced_text += ' ' * spacing
        
        return spaced_text
    
    def _load_font(self, font_family: str, font_size: int):
        """Load font with comprehensive fallbacks"""
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