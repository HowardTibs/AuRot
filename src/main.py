"""
Simplified Gradio application with LIVE PREVIEW, Vignette, Dark Theme, SCRIPT INPUT, and REDDIT INTEGRATION
NO FADE EFFECTS - Text appears instantly and stays visible
Preview matches final output exactly
Two-audio system: Title audio + Main story audio with 1-second delay
REFINED VERSION with enhanced Reddit integration
FIXED: Live preview outline effect (no word duplication)
UPDATED: Added 3.5-second ending silence with last text remaining visible
FIXED: Video duration preserved for dead air (no cutting)
FIXED: Dynamic font sizing and permanent margins for 1-5 words per segment
"""

import gradio as gr
import tempfile
import logging
import threading
import time
import re
import cv2  # For video verification
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

# Import our professional modules
from audio_processor import AudioProcessor
from text_sync import TextSynchronizer
from video_processor import ProVideoProcessor, cache_manager
from reddit_generator import RedditTemplateGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScriptMatcher:
    """Handles matching provided script with Whisper timing data while preserving formatting"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def clean_text_for_matching(self, text: str) -> str:
        """Clean text for comparison only (remove punctuation, lowercase)"""
        # Remove punctuation, convert to lowercase, normalize whitespace
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def preprocess_script(self, script: str) -> str:
        """Preprocess script to remove dialogue lines and clean formatting"""
        lines = script.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines starting with "-" (dialogue indicators)
            if line and not line.startswith('-'):
                processed_lines.append(line)
        
        return ' '.join(processed_lines)
    
    def split_script_into_words(self, script: str) -> List[Dict]:
        """Split script into words while preserving original formatting"""
        # Preprocess to remove dialogue lines
        processed_script = self.preprocess_script(script)
        
        # Split into words while keeping punctuation and capitalization
        # Use regex to split on whitespace but keep the words with their punctuation
        words_with_formatting = re.findall(r'\S+', processed_script)
        
        # Create word objects with both original and cleaned versions
        script_words = []
        for word in words_with_formatting:
            script_words.append({
                'original': word,  # Keep original with punctuation/caps
                'cleaned': self.clean_text_for_matching(word)  # For matching
            })
        
        return script_words
    
    def match_script_to_timing(self, script: str, whisper_words: List[Dict]) -> List[Dict]:
        """
        Match provided script words to Whisper timing data while preserving formatting
        
        Args:
            script: User-provided original script
            whisper_words: Whisper transcription with timing
            
        Returns:
            List of word dictionaries with accurate text and timing
        """
        if not script.strip():
            return whisper_words
        
        script_words = self.split_script_into_words(script)
        whisper_clean = [self.clean_text_for_matching(w.get('text', '')) for w in whisper_words]
        
        logger.info(f"Matching {len(script_words)} script words to {len(whisper_words)} whisper words")
        
        # Advanced alignment - try to match script words to whisper timing
        matched_words = []
        script_index = 0
        
        for i, whisper_word in enumerate(whisper_words):
            whisper_clean_text = self.clean_text_for_matching(whisper_word.get('text', ''))
            
            if script_index < len(script_words):
                script_word_obj = script_words[script_index]
                script_word_cleaned = script_word_obj['cleaned']
                script_word_original = script_word_obj['original']
                
                # Check if words match (with some fuzzy matching)
                if self.words_match(script_word_cleaned, whisper_clean_text):
                    # Use original formatted script word with whisper timing
                    matched_word = {
                        'text': script_word_original,  # Use original with punctuation!
                        'start': whisper_word.get('start', 0),
                        'end': whisper_word.get('end', 0),
                        'confidence': whisper_word.get('confidence', 0.9),
                        'source': 'script_matched'
                    }
                    matched_words.append(matched_word)
                    script_index += 1
                else:
                    # Use whisper word if no match found (keeps Whisper's smart formatting)
                    matched_words.append({
                        **whisper_word,
                        'source': 'whisper_fallback'
                    })
            else:
                # Use remaining whisper words if script is exhausted
                matched_words.append({
                    **whisper_word,
                    'source': 'whisper_remaining'
                })
        
        # Add any remaining script words with estimated timing
        while script_index < len(script_words):
            last_time = matched_words[-1]['end'] if matched_words else 0
            estimated_duration = 0.5  # 500ms per word estimate
            
            script_word_obj = script_words[script_index]
            matched_words.append({
                'text': script_word_obj['original'],  # Use original formatting
                'start': last_time,
                'end': last_time + estimated_duration,
                'confidence': 0.8,
                'source': 'script_estimated'
            })
            script_index += 1
        
        logger.info(f"Script matching complete: {len(matched_words)} final words")
        return matched_words
    
    def words_match(self, word1: str, word2: str) -> bool:
        """Check if two words match with fuzzy matching"""
        # Exact match
        if word1 == word2:
            return True
        
        # Check if one word contains the other
        if word1 in word2 or word2 in word1:
            return True
        
        # Simple edit distance for short words
        if len(word1) <= 3 or len(word2) <= 3:
            return self.simple_edit_distance(word1, word2) <= 1
        
        # For longer words, allow more variation
        return self.simple_edit_distance(word1, word2) <= 2
    
    def simple_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate simple edit distance between two strings"""
        if len(s1) < len(s2):
            return self.simple_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class SimplifiedReelMakerApp:
    """Simplified app with NO FADE EFFECTS + Reddit Integration + 3.5s Dead Air"""
    
    def __init__(self):
        """Initialize all processors"""
        logger.info("Initializing Simplified Reel Maker Application - NO FADE EFFECTS + Reddit Integration + 3.5s Dead Air")
        
        try:
            self.audio_processor = AudioProcessor(model_size="base")
            self.text_synchronizer = TextSynchronizer()
            self.video_processor = ProVideoProcessor()
            self.script_matcher = ScriptMatcher()
            self.reddit_generator = RedditTemplateGenerator()
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
            # Dead air configuration
            self.ending_silence_duration = 3.5  # 3.5 seconds of dead air
            
            logger.info("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def _reconstruct_formatted_text(self, words: List[Dict]) -> str:
        """Reconstruct text from words while preserving proper formatting"""
        if not words:
            return ""
        
        reconstructed = []
        for word_info in words:
            word_text = word_info.get('text', '')
            reconstructed.append(word_text)
        
        # Join words with spaces, but handle punctuation smartly
        text = ' '.join(reconstructed)
        
        # Clean up extra spaces around punctuation
        # Remove space before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        # Remove space before closing punctuation
        text = re.sub(r'\s+([)\]}])', r'\1', text)
        # Remove space after opening punctuation  
        text = re.sub(r'([(\[{])\s+', r'\1', text)
        
        return text
    
    def get_title_audio_duration(self, title_audio_file: str) -> float:
        """Get duration of title audio file"""
        try:
            return self.audio_processor.get_audio_duration(title_audio_file)
        except Exception as e:
            logger.error(f"Failed to get title audio duration: {e}")
            return 0.0
    
    def process_reel_with_reddit(self, 
                                title_audio_file: str,
                                main_audio_file: str,
                                video_file: str,
                                reddit_image,  # PIL Image from reddit generator
                                original_script: str,
                                font_family: str,
                                font_size: int,
                                text_color: str,
                                stroke_color: str,
                                stroke_width: int,
                                position: str,
                                text_case: str,
                                vignette_strength: float,
                                words_per_segment: int) -> Tuple[Optional[str], str]:
        """
        FIXED: Process reel with title audio + reddit image + main story audio + 3.5s dead air
        """
        try:
            # Cache management - clean up previous generation
            logger.info("🗑️ Cleaning up previous generation...")
            self.video_processor.cleanup_previous_session()
            
            # Enhanced input validation
            if not title_audio_file or not main_audio_file or not video_file:
                return None, "❌ Please upload title audio, main audio, and video files"
            
            if reddit_image is None:
                return None, "❌ Please generate a Reddit post image first"
            
            logger.info("🎨 Starting Reddit-integrated reel processing with 3.5s dead air...")
            
            # Step 1: Validate dual audio system
            status = "🎵 Validating dual audio system...\n"
            
            is_valid, validation_msg = self.audio_processor.validate_dual_audio_files(
                title_audio_file, main_audio_file
            )
            
            if not is_valid:
                return None, status + validation_msg
            
            status += validation_msg + "\n"
            
            # Step 2: Process dual audio system with ending silence
            status += "⚙️ Processing dual audio timing with 3.5s ending silence...\n"
            
            dual_audio_result = self.audio_processor.process_dual_audio_system(
                title_audio_file, main_audio_file, self.ending_silence_duration
            )
            
            if not dual_audio_result['success']:
                return None, status + dual_audio_result['error']
            
            # Extract timing information
            title_duration = dual_audio_result['title_duration']
            main_duration = dual_audio_result['main_duration']
            delay_duration = dual_audio_result['delay_duration']
            reddit_display_duration = dual_audio_result['reddit_display_duration']
            ending_silence = dual_audio_result['ending_silence']
            total_duration = dual_audio_result['total_duration']
            main_transcription = dual_audio_result['main_transcription']
            
            status += f"✅ Dual audio processed with ending silence:\n"
            status += f"   • Title: {title_duration:.2f}s\n"
            status += f"   • Delay: {delay_duration:.2f}s\n"
            status += f"   • Reddit Display: {reddit_display_duration:.2f}s\n"
            status += f"   • Main Story: {main_duration:.2f}s\n"
            status += f"   • Audio Content Ends: {dual_audio_result['audio_content_duration']:.2f}s\n"
            status += f"   • Ending Silence: {ending_silence:.2f}s\n"
            status += f"   • Total Duration: {total_duration:.2f}s\n"
            
            # Log timing verification
            logger.info(f"🕐 TIMING VERIFICATION:")
            logger.info(f"  • Audio content duration: {dual_audio_result['audio_content_duration']:.2f}s")
            logger.info(f"  • Dead air starts at: {dual_audio_result['dead_air_start']:.2f}s")
            logger.info(f"  • Total video duration: {total_duration:.2f}s")
            logger.info(f"  • Dead air duration: {ending_silence:.2f}s")
            
            # Step 3: Script Matching (if provided)
            if original_script and original_script.strip():
                status += "📝 Matching script to audio timing...\n"
                
                matched_words = self.script_matcher.match_script_to_timing(
                    original_script, main_transcription['words']
                )
                
                main_transcription['words'] = matched_words
                main_transcription['text'] = self._reconstruct_formatted_text(matched_words)
                
                script_matched = sum(1 for w in matched_words if w.get('source') == 'script_matched')
                total_words = len(matched_words)
                
                status += f"✅ Script matching: {script_matched}/{total_words} words matched\n"
            else:
                status += "📝 Using Whisper transcription (no script provided)\n"
            
            # Step 4: Prepare combined audio with ending silence
            status += "🎵 Preparing combined audio with ending silence...\n"
            
            combined_audio_path = self.audio_processor.prepare_dual_audio_for_video(
                title_audio_file, main_audio_file, delay_duration, ending_silence
            )
            
            status += "✅ Combined audio prepared with ending silence\n"
            
            # Step 5: Create text segments with offset timing and extended last segment
            status += "📝 Creating text segments with extended last segment for dead air...\n"
            
            self.text_synchronizer.words_per_display = words_per_segment
            text_segments = self.text_synchronizer.create_display_segments_with_offset_and_extended_end(
                main_transcription['words'], 
                total_duration,  # Use exact total duration
                reddit_display_duration,
                ending_silence
            )
            
            if not text_segments:
                return None, status + "❌ Failed to create text segments"
            
            # VALIDATION: Verify dead air text extension
            if text_segments:
                last_segment = text_segments[-1]
                if last_segment.get('has_dead_air_extension', False):
                    status += f"🔥 VERIFIED: Last text segment extended for dead air\n"
                    status += f"   • Text: '{last_segment['text'][:40]}{'...' if len(last_segment['text']) > 40 else ''}'\n"
                    status += f"   • Will remain visible until: {last_segment['end']:.2f}s\n"
                    status += f"   • Dead air period: {last_segment.get('dead_air_start', 0):.2f}s - {total_duration:.2f}s\n"
                else:
                    status += f"⚠️ WARNING: Last text segment NOT extended for dead air\n"
            
            status += f"✅ Created {len(text_segments)} text segments with extended last segment\n"
            
            # Step 6: Process background video for FULL duration including dead air
            status += f"🎬 Processing background video for full duration including dead air (vignette: {vignette_strength:.1f})...\n"
            
            processed_video_path = self.video_processor.process_background_video(
                video_file, total_duration, vignette_strength  # Exact total duration
            )
            
            status += "✅ Background video processed\n"
            
            # Step 7: Create text overlays
            status += "✨ Creating text overlays with dead air support...\n"
            
            style_config = {
                'font_family': font_family,
                'font_size': font_size,
                'text_color': text_color,
                'stroke_color': stroke_color,
                'stroke_width': stroke_width,
                'position': position,
                'text_case': text_case,
                'words_per_segment': words_per_segment  # ADDED: Pass words per segment for dynamic sizing
            }
            
            text_overlays = self.video_processor.create_text_overlays(text_segments, style_config)
            
            status += f"✅ Created {len(text_overlays)} text overlays\n"
            
            # Step 8: Compose video with Reddit integration
            status += "🎭 Composing video with Reddit integration...\n"
            
            video_with_reddit_and_text = self.video_processor.compose_video_with_reddit(
                processed_video_path, text_overlays, reddit_image, reddit_display_duration
            )
            
            status += "✅ Reddit + text composition complete\n"
            
            # Step 9: Add prepared audio with ending silence (FIXED: preserves video length)
            status += "🔊 Adding prepared audio while preserving video length for dead air...\n"
            
            video_with_audio = self.video_processor.add_prepared_audio_track(
                video_with_reddit_and_text, combined_audio_path
            )
            
            status += "✅ Audio with ending silence added (video length preserved)\n"
            
            # Step 10: Export final video
            status += "💾 Exporting final video...\n"
            
            final_video_path = self.video_processor.export_video(video_with_audio)
            
            # Verify final video duration
            try:
                verify_cap = cv2.VideoCapture(final_video_path)
                if verify_cap.isOpened():
                    verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                    verify_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    final_duration = verify_frame_count / verify_fps if verify_fps > 0 else 0
                    verify_cap.release()
                    
                    logger.info(f"🎬 FINAL VIDEO VERIFICATION:")
                    logger.info(f"  • Expected duration: {total_duration:.2f}s")
                    logger.info(f"  • Actual duration: {final_duration:.2f}s")
                    logger.info(f"  • Dead air period: {dual_audio_result['dead_air_start']:.2f}s - {total_duration:.2f}s")
                    
                    if final_duration >= total_duration - 0.5:
                        status += f"✅ Final video verification: {final_duration:.2f}s duration (includes dead air)\n"
                    else:
                        status += f"⚠️ Video duration warning: {final_duration:.2f}s (expected {total_duration:.2f}s)\n"
                else:
                    status += "✅ Video exported successfully\n"
            except Exception as e:
                logger.warning(f"Could not verify final video duration: {e}")
                status += "✅ Video exported successfully\n"
            
            # Create comprehensive final status
            script_info = ""
            if original_script and original_script.strip():
                matched_words = main_transcription['words']
                script_matched = sum(1 for w in matched_words if w.get('source') == 'script_matched')
                total_words = len(matched_words)
                script_info = f"\n🎯 **Script Integration:**\n   • {script_matched}/{total_words} words matched\n   • Preserved original formatting and punctuation"
            
            # Get device info for status
            device_info = self.audio_processor.get_device_info()
            device_status = f"🖥️ **Processing Device:** {device_info['device'].upper()}"
            if device_info['device'] == 'cuda':
                device_status += f" ({device_info.get('gpu_name', 'GPU')})"
            
            final_status = status + f"""
🎉 **Reddit-Style Reel Complete with 3.5s Dead Air!**

📊 **Video Composition:**
• **Reddit Phase:** {reddit_display_duration:.2f}s (title: {title_duration:.2f}s + delay: {delay_duration:.2f}s)
• **Story Phase:** {main_duration:.2f}s with synchronized text
• **Dead Air Phase:** {ending_silence:.2f}s (last text remains visible)
• **Total Duration:** {total_duration:.2f}s
• **Resolution:** 1080x1920 (9:16 aspect ratio)
• **Text Segments:** {len(text_segments)} ({words_per_segment} words each)
• **Font:** {font_family} ({font_size}px with dynamic sizing)
• **Vignette:** {vignette_strength:.1f} strength
• **Animation:** Heartbeat effect (90% slower)
• **Display Mode:** INSTANT text (no fade effects)
• **Margins:** Permanent 10% margins maintained

🎵 **Audio System:**
• **Dual Track:** Title + Main story + {ending_silence:.2f}s silence
• **Synchronization:** Perfect timing alignment
• **Quality:** 128kbps AAC

⏱️ **Dead Air Feature:**
• **Duration:** {ending_silence:.2f} seconds at the end
• **Behavior:** Last text remains visible, background continues
• **Purpose:** Reflection time for viewers

{device_status}

{script_info}

📝 **Story Preview:**
{main_transcription['text'][:200]}{'...' if len(main_transcription['text']) > 200 else ''}

✨ Your Reddit-integrated reel with dead air is ready for social media!
            """
            
            logger.info("🎨 Reddit reel processing with dead air completed successfully")
            return final_video_path, final_status
            
        except Exception as e:
            error_message = f"""❌ **Processing Error:** {str(e)}

🔧 **Troubleshooting Steps:**
1. **Verify Files:** Ensure all uploaded files are valid
2. **Reddit Image:** Generate Reddit post image first
3. **Audio Quality:** Use clear audio files (MP3/WAV)
4. **Video Format:** Use MP4 format for best compatibility
5. **File Sizes:** Keep files under reasonable sizes

📞 **Support Tips:**
• Title audio: 0.5s - 30s duration
• Main audio: 1s - 5 minutes duration  
• Video: MP4 format recommended
• Reddit image: Generate before creating reel

🔄 **Quick Fix:** Try generating Reddit post again and re-upload audio files
            """
            logger.error(f"Reddit reel processing failed: {e}")
            return None, error_message

def calculate_dynamic_font_size(base_font_size: int, words_per_segment: int, is_preview: bool = False) -> int:
    """
    FIXED: Calculate dynamic font size based on words per segment to maintain readability and margins
    
    Args:
        base_font_size: Original font size setting
        words_per_segment: Number of words per segment (1-5)
        is_preview: Whether this is for preview (different scaling)
        
    Returns:
        Adjusted font size
    """
    # Dynamic scaling factors for different word counts
    scaling_factors = {
        1: 1.0,    # 1 word - full size
        2: 0.95,   # 2 words - slightly smaller
        3: 0.85,   # 3 words - moderate reduction
        4: 0.75,   # 4 words - significant reduction
        5: 0.65    # 5 words - most reduction
    }
    
    scale_factor = scaling_factors.get(words_per_segment, 0.65)
    adjusted_size = int(base_font_size * scale_factor)
    
    # Apply preview scaling if needed
    if is_preview:
        adjusted_size = max(8, adjusted_size // 4)  # Preview scaling
    else:
        adjusted_size = max(20, adjusted_size)  # Minimum readable size for video
    
    return adjusted_size

def create_live_preview(font_family, font_size, text_color, stroke_color, stroke_width, position, text_case, vignette_strength, words_per_segment):
    """FIXED: Create live preview with INSTANT display, dynamic font sizing, and permanent margins for 1-5 words"""
    
    # Sample text for preview based on words per segment with better examples
    sample_texts = {
        1: "Amazing",
        2: "This is",
        3: "This is incredible",
        4: "This story will blow",
        5: "This incredible story will absolutely"
    }
    
    sample_text = sample_texts.get(words_per_segment, "This incredible story will absolutely")
    
    # Apply text case
    if text_case == 'uppercase':
        display_text = sample_text.upper()
    elif text_case == 'lowercase':
        display_text = sample_text.lower()
    else:
        display_text = sample_text
    
    # FIXED: Calculate dynamic font size based on words per segment
    dynamic_font_size = calculate_dynamic_font_size(font_size, words_per_segment, is_preview=True)
    
    # Create CSS styles - EXACT SAME AS VIDEO OUTPUT
    font_weight = 'bold' if 'Bold' in font_family else 'normal'
    clean_font = font_family.replace(' Bold', '')
    
    # Position styling
    position_styles = {
        'top': 'align-items: flex-start; padding-top: 15%;',
        'center': 'align-items: center;',
        'bottom': 'align-items: flex-end; padding-bottom: 15%;'
    }
    
    # Create vignette effect
    vignette_opacity = vignette_strength * 0.7  # Scale down for preview
    
    # FIXED: Dynamic stroke width scaling based on font size
    preview_stroke_width = max(0.3, (stroke_width * dynamic_font_size) / (font_size * 4))
    
    # FIXED: Add text length indicator for overflow prevention
    text_length_class = "normal"
    if len(display_text) > 25:
        text_length_class = "long"
    elif len(display_text) > 35:
        text_length_class = "very-long"
    
    preview_html = f"""
    <div style="
        width: 270px;
        height: 480px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        display: flex;
        justify-content: center;
        {position_styles.get(position, position_styles['center'])}
        padding: 20px;
        box-sizing: border-box;
        margin: 0 auto;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
        border: 2px solid #2d3748;
    ">
        <!-- Vignette overlay -->
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at center, transparent 0%, transparent 30%, rgba(0,0,0,{vignette_opacity}) 70%);
            pointer-events: none;
            border-radius: 10px;
        "></div>
        
        <div style="
            position: absolute;
            top: 10px;
            right: 15px;
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            color: white;
            font-weight: bold;
            z-index: 10;
        ">9:16 Preview</div>
        
        <div style="
            position: absolute;
            top: 10px;
            left: 15px;
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 9px;
            color: white;
            font-weight: bold;
            z-index: 10;
        ">{words_per_segment} word{'s' if words_per_segment != 1 else ''}</div>
        
        <div style="
            position: absolute;
            bottom: 15px;
            left: 15px;
            background: rgba(0,255,0,0.2);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 8px;
            color: #90EE90;
            font-weight: bold;
            z-index: 10;
        ">+3.5s dead air</div>
        
        <div style="
            position: absolute;
            bottom: 15px;
            right: 15px;
            background: rgba(255,165,0,0.2);
            padding: 3px 6px;
            border-radius: 8px;
            font-size: 7px;
            color: #FFA500;
            font-weight: bold;
            z-index: 10;
        ">Font: {dynamic_font_size}px</div>
        
        <div id="preview-text" style="
            font-family: '{clean_font}', Arial, sans-serif;
            font-size: {dynamic_font_size}px;
            font-weight: {font_weight};
            color: {text_color};
            text-align: center;
            -webkit-text-stroke: {preview_stroke_width}px {stroke_color};
            text-shadow: 0px 0px 3px rgba(0,0,0,0.5);
            max-width: 80%;
            word-wrap: break-word;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 5;
            position: relative;
            animation: heartbeat 5.26s ease-in-out infinite;
            line-height: 1.1;
            padding: 0 8%;
            overflow-wrap: break-word;
            hyphens: auto;
        ">{display_text}</div>
    </div>
    
    <style>
    @keyframes heartbeat {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.03); }}
    }}
    </style>
    """
    
    return preview_html

def generate_reddit_post(title, username, avatar, likes, comments, use_saved_profile, save_current_profile, app_instance):
    """Generate Reddit post template"""
    if not title.strip():
        return None, "Please enter a title for the post"
    
    # FIXED: Set default comments to 99 if empty
    if not comments or comments.strip() == "":
        comments = "99"
    
    # Generate the image
    img = app_instance.reddit_generator.create_reddit_template(
        title=title,
        username=username if username.strip() else None,
        avatar_image=avatar,
        likes=likes if likes.strip() else "99+",
        comments=comments,
        use_saved_profile=use_saved_profile
    )
    
    # Save profile if requested
    status_message = ""
    if save_current_profile:
        save_msg = app_instance.reddit_generator.save_profile(username, avatar)
        status_message = save_msg
    
    # Get profile info
    profile_info = app_instance.reddit_generator.get_saved_profile_info()
    
    return img, f"{status_message}\n\n{profile_info}"

def clear_reddit_inputs():
    """Clear all Reddit input fields"""
    return "", "", None, "", "", False, False

def create_interface() -> gr.Blocks:
    """Create interface with NO FADE EFFECTS + Reddit Integration + 3.5s Dead Air + Fixed Dynamic Font Sizing"""
    
    # Initialize the app
    app = SimplifiedReelMakerApp()
    
    # Professional dark theme CSS
    professional_css = """
    /* Professional Dark Theme */
    .gradio-container {
        background-color: #0f1419 !important;
        color: #ffffff !important;
    }
    
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 30px;
        border: 2px solid #2d3748;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .section-header {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
        color: #ffffff !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4299e1;
        margin: 20px 0;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .section-header h3 {
        color: #ffffff !important;
        margin: 0 !important;
        font-size: 18px !important;
    }
    
    .preview-container {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid #4a5568;
        margin: 20px 0;
    }
    
    .controls-container {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid #4a5568;
        margin: 20px 0;
    }
    
    .script-container {
        background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid #3182ce;
        margin: 20px 0;
    }
    
    .reddit-container {
        background: linear-gradient(135deg, #ff4500 0%, #ff6500 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 2px solid #ff4500;
        margin: 20px 0;
    }
    
    /* Dark theme for inputs */
    .gr-input, .gr-textarea, .gr-dropdown, .gr-slider input {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
    }
    
    .gr-label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .gr-button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .performance-info {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
        border-radius: 12px;
        color: white;
        border: 2px solid #3182ce;
        box-shadow: 0 5px 15px rgba(43, 108, 176, 0.3);
    }
    
    .tab-nav button {
        background: linear-gradient(45deg, #2d3748, #4a5568) !important;
        color: white !important;
        border: 1px solid #4a5568 !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    """
    
    with gr.Blocks(
        title="Professional Reel Maker with Reddit Integration + Dead Air + Fixed Dynamic Font Sizing",
        css=professional_css,
        theme=gr.themes.Base()
    ) as interface:
        
        # Professional header
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 Professional Reel Maker + Reddit Integration + Dead Air</h1>
            <p>Create engaging vertical videos with Reddit-style posts and AI-powered transcription</p>
            <div style="margin-top: 15px;">
                <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🎨 Professional Grade</span>
                <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">📱 Live Preview</span>
                <span style="background: linear-gradient(45deg, #ff4500, #ff6500); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🔥 Reddit Posts</span>
                <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🎵 Dual Audio</span>
                <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">⚡ INSTANT Display</span>
                <span style="background: linear-gradient(45deg, #32cd32, #228b22); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">⏱️ 3.5s Dead Air</span>
                <span style="background: linear-gradient(45deg, #ff6b35, #f7931e); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🔤 Dynamic Font Sizing</span>
            </div>
        </div>
        """)
        
        # Create tabbed interface
        with gr.Tabs():
            
            # TAB 1: REDDIT POST GENERATOR
            with gr.TabItem("🔥 Reddit Post Generator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>✏️ Post Content</h3></div>')
                        
                        with gr.Group(elem_classes="reddit-container"):
                            reddit_title_input = gr.Textbox(
                                label="📄 Post Title",
                                placeholder="Enter your Reddit post title here...",
                                lines=3,
                                max_lines=5
                            )
                        
                        gr.HTML('<div class="section-header"><h3>👤 User Profile</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            reddit_username_input = gr.Textbox(
                                label="Username",
                                placeholder="Enter username (leave empty to use saved profile)",
                                value=""
                            )
                            
                            reddit_avatar_input = gr.File(
                                label="Avatar Image",
                                file_types=["image"],
                                type="filepath"
                            )
                        
                        gr.HTML('<div class="section-header"><h3>📊 Post Stats</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            with gr.Row():
                                reddit_likes_input = gr.Textbox(
                                    label="Likes",
                                    placeholder="99+",
                                    value="99+",
                                    scale=1
                                )
                                reddit_comments_input = gr.Textbox(
                                    label="Comments", 
                                    placeholder="99+",
                                    value="99+",
                                    scale=1
                                )
                        
                        gr.HTML('<div class="section-header"><h3>⚙️ Profile Settings</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            reddit_use_saved_profile = gr.Checkbox(
                                label="Use Saved Profile",
                                value=False,
                                info="Use previously saved username and avatar"
                            )
                            
                            reddit_save_current_profile = gr.Checkbox(
                                label="Save Current Profile",
                                value=False,
                                info="Save current username and avatar for future use"
                            )
                        
                        with gr.Row():
                            reddit_generate_btn = gr.Button("🚀 Generate Reddit Post", variant="primary", scale=2)
                            reddit_clear_btn = gr.Button("🗑️ Clear", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>🖼️ Generated Reddit Post</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            reddit_output_image = gr.Image(
                                label="Reddit Post",
                                type="pil",
                                interactive=False
                            )
                            
                            reddit_status_output = gr.Textbox(
                                label="Status & Profile Info",
                                interactive=False,
                                lines=4
                            )
                        
                        gr.Markdown("### 💾 Download")
                        gr.Markdown("Right-click on the generated image above to save it!")
                        gr.Markdown("### ⚠️ Important")
                        gr.Markdown("**Generate your Reddit post first** before creating the reel in the next tab!")
            
            # TAB 2: REDDIT REEL GENERATOR - Upload, Process & Results
            with gr.TabItem("🎬 Generate Reddit Reel"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>🎵 Upload Your Audio Files</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            title_audio_input = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🔥 Title Audio (displays Reddit post)",
                                show_download_button=False
                            )
                            
                            main_audio_input = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎵 Main Story Audio (displays text overlay)",
                                show_download_button=False
                            )
                            
                            video_input = gr.Video(
                                sources=["upload"],
                                label="🎬 Background Video",
                                height=200
                            )
                        
                        # Script Input Section
                        gr.HTML('<div class="section-header"><h3>📝 Original Script (Optional)</h3></div>')
                        
                        with gr.Group(elem_classes="script-container"):
                            original_script = gr.Textbox(
                                label="📄 Original Script/Transcript",
                                placeholder="Paste your original script here for improved accuracy...\n\nExample:\nHello everyone, welcome to my channel!\nToday we're going to talk about AI technology.\nIt's absolutely fascinating how far we've come.\n\nNote: Lines starting with '-' (dialogue indicators) will be automatically removed.",
                                lines=6,
                                value="",
                                info="Optional: Provide the original script to improve text accuracy. Punctuation, capitalization, and formatting will be preserved!"
                            )
                        
                        # Professional Process button
                        process_btn = gr.Button(
                            "🎬 Create Reddit Reel (Title + Story + 3.5s Dead Air)",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Performance info
                        gr.HTML("""
                        <div class="performance-info">
                            <strong>🚀 Reddit Reel Features + 3.5s Dead Air + Dynamic Font Sizing</strong><br>
                            🔥 Reddit post display during title • 🎵 Dual audio system • ⏱️ 1-second delay<br>
                            🎨 Live preview • 📱 9:16 aspect ratio • 🌅 Background vignette<br>
                            🔤 Smart text wrapping • 🤖 AI transcription • 📝 Script matching<br>
                            ⚡ INSTANT text display (NO fade effects) • 💓 Heartbeat animation (90% slower)<br>
                            <strong>🔤 NEW: Dynamic font sizing for 1-5 words per segment</strong><br>
                            <strong>📐 NEW: Permanent 10% margins to prevent text overflow</strong><br>
                            <strong>⏱️ NEW: 3.5-second dead air at end with last text remaining visible</strong>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>📱 Your Reddit Reel</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            video_output = gr.Video(
                                label="Reddit-Style Reel with Dead Air",
                                height=400,
                                show_download_button=True
                            )
                            
                            status_output = gr.Textbox(
                                label="Real-time Processing Status",
                                lines=12,
                                interactive=False
                            )
            
            # TAB 3: STYLE & PREVIEW - All Controls with Live Preview
            with gr.TabItem("🎨 Style & Preview"):
                gr.HTML('<div class="section-header"><h3>🎨 Text Styling with Live Preview (INSTANT DISPLAY + Dynamic Font Sizing + Dead Air)</h3></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>✨ Text Controls</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            font_family = gr.Dropdown(
                                choices=[
                                    "Arial",
                                    "Arial Bold",
                                    "Impact",
                                    "Times New Roman",
                                    "Helvetica",
                                    "Georgia",
                                    "Trebuchet MS",
                                    "Verdana"
                                ],
                                value="Arial Bold",
                                label="Font Family"
                            )
                            
                            with gr.Row():
                                font_size = gr.Slider(
                                    minimum=30,
                                    maximum=120,
                                    value=60,
                                    step=5,
                                    label="Base Font Size (auto-adjusts for word count)",
                                    info="Larger word counts automatically use smaller fonts"
                                )
                                
                                text_case = gr.Dropdown(
                                    choices=[
                                        ("Regular", "regular"),
                                        ("UPPERCASE", "uppercase"),
                                        ("lowercase", "lowercase")
                                    ],
                                    value="regular",
                                    label="Text Case"
                                )
                            
                            with gr.Row():
                                text_color = gr.ColorPicker(
                                    value="#FFFFFF",
                                    label="Text Color"
                                )
                                
                                stroke_color = gr.ColorPicker(
                                    value="#000000",
                                    label="Outline Color"
                                )
                            
                            stroke_width = gr.Slider(
                                minimum=0,
                                maximum=15,
                                value=3,
                                step=1,
                                label="Outline Width"
                            )
                            
                            position = gr.Dropdown(
                                choices=["top", "center", "bottom"],
                                value="center",
                                label="Text Position"
                            )
                        
                        gr.HTML('<div class="section-header"><h3>🌅 Video Effects</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            vignette_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                label="Vignette Strength",
                                info="Dims background edges to highlight text"
                            )
                            
                            words_per_segment = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=2,
                                step=1,
                                label="📝 Words per text segment (1-5)",
                                info="Font size automatically adjusts - more words = smaller font"
                            )
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>👁️ Live Preview (INSTANT DISPLAY + Dynamic Font Sizing + Dead Air)</h3></div>')
                        
                        with gr.Group(elem_classes="preview-container"):
                            live_preview = gr.HTML(
                                create_live_preview(
                                    "Arial Bold", 60, "#FFFFFF", "#000000", 3, 
                                    "center", "regular", 0.3, 2
                                ),
                                label="Live Text Preview"
                            )
                            
                            gr.HTML("""
                            <div style="text-align: center; margin-top: 20px; color: #a0aec0; font-size: 14px;">
                                <strong>🎯 Preview Features (INSTANT DISPLAY + Dynamic Font Sizing + Dead Air):</strong><br>
                                • Real-time font changes with dynamic sizing<br>
                                • Accurate 9:16 aspect ratio<br>
                                • Live color updates<br>
                                • Position preview<br>
                                • Vignette effect preview<br>
                                • Heartbeat animation (90% slower)<br>
                                • <strong>NEW: Font automatically adjusts for 1-5 words</strong><br>
                                • <strong>NEW: Permanent 10% margins maintained</strong><br>
                                • <strong>NO FADE EFFECTS - Text appears instantly!</strong><br>
                                • <strong>EXACTLY MATCHES FINAL OUTPUT!</strong><br>
                                • <strong>FIXED: Proper outline (no duplication)!</strong><br>
                                • <strong>NEW: 3.5s dead air with last text visible!</strong>
                            </div>
                            """)
        
        # Update live preview when any styling option changes
        style_inputs = [
            font_family, font_size, text_color, stroke_color, stroke_width,
            position, text_case, vignette_strength, words_per_segment
        ]
        
        for input_component in style_inputs:
            input_component.change(
                fn=create_live_preview,
                inputs=style_inputs,
                outputs=[live_preview]
            )
        
        # Reddit post generation
        reddit_generate_btn.click(
            fn=lambda *args: generate_reddit_post(*args, app),
            inputs=[
                reddit_title_input, reddit_username_input, reddit_avatar_input, 
                reddit_likes_input, reddit_comments_input, reddit_use_saved_profile, 
                reddit_save_current_profile
            ],
            outputs=[reddit_output_image, reddit_status_output]
        )
        
        reddit_clear_btn.click(
            fn=clear_reddit_inputs,
            outputs=[reddit_title_input, reddit_username_input, reddit_avatar_input, 
                    reddit_likes_input, reddit_comments_input, reddit_use_saved_profile, 
                    reddit_save_current_profile]
        )
        
        # Connect processing function with Reddit integration
        process_btn.click(
            fn=app.process_reel_with_reddit,
            inputs=[
                title_audio_input,
                main_audio_input,
                video_input,
                reddit_output_image,  # Pass the generated Reddit image
                original_script,
                font_family,
                font_size,
                text_color,
                stroke_color,
                stroke_width,
                position,
                text_case,
                vignette_strength,
                words_per_segment
            ],
            outputs=[video_output, status_output]
        )
        
        # Load profile info on startup
        interface.load(
            fn=lambda: app.reddit_generator.get_saved_profile_info(),
            outputs=reddit_status_output
        )
    
    return interface

def main():
    """Main entry point"""
    logger.info("🎬 Starting Professional Reel Maker with Reddit Integration + 3.5s Dead Air + Fixed Dynamic Font Sizing")
    
    # Test OpenCV installation
    try:
        import cv2
        logger.info(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed. Please run: pip install opencv-python")
        return
    
    try:
        # Create and launch interface
        interface = create_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()