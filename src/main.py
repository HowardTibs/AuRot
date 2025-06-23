"""
Simplified Gradio application with LIVE PREVIEW, Vignette, Dark Theme, and SCRIPT INPUT
Clean interface with tabbed layout, real-time preview, and optional script input for accuracy
"""

import gradio as gr
import tempfile
import logging
import threading
import time
import re
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

# Import our professional modules
from audio_processor import AudioProcessor
from text_sync import TextSynchronizer
from video_processor import ProVideoProcessor, cache_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScriptMatcher:
    """Handles matching provided script with Whisper timing data"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def clean_text(self, text: str) -> str:
        """Clean text for comparison"""
        # Remove punctuation, convert to lowercase, normalize whitespace
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def split_script_into_words(self, script: str) -> List[str]:
        """Split script into individual words"""
        cleaned_script = self.clean_text(script)
        return cleaned_script.split()
    
    def match_script_to_timing(self, script: str, whisper_words: List[Dict]) -> List[Dict]:
        """
        Match provided script words to Whisper timing data
        
        Args:
            script: User-provided original script
            whisper_words: Whisper transcription with timing
            
        Returns:
            List of word dictionaries with accurate text and timing
        """
        if not script.strip():
            return whisper_words
        
        script_words = self.split_script_into_words(script)
        whisper_clean = [self.clean_text(w.get('text', '')) for w in whisper_words]
        
        logger.info(f"Matching {len(script_words)} script words to {len(whisper_words)} whisper words")
        
        # Simple alignment - try to match script words to whisper timing
        matched_words = []
        script_index = 0
        
        for i, whisper_word in enumerate(whisper_words):
            whisper_clean_text = self.clean_text(whisper_word.get('text', ''))
            
            if script_index < len(script_words):
                script_word = script_words[script_index]
                
                # Check if words match (with some fuzzy matching)
                if self.words_match(script_word, whisper_clean_text):
                    # Use script word with whisper timing
                    matched_word = {
                        'text': script_word,
                        'start': whisper_word.get('start', 0),
                        'end': whisper_word.get('end', 0),
                        'confidence': whisper_word.get('confidence', 0.9),
                        'source': 'script_matched'
                    }
                    matched_words.append(matched_word)
                    script_index += 1
                else:
                    # Use whisper word if no match found
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
            
            matched_words.append({
                'text': script_words[script_index],
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
    """Simplified app with live preview and script input"""
    
    def __init__(self):
        """Initialize all processors"""
        logger.info("Initializing Simplified Reel Maker Application")
        
        try:
            self.audio_processor = AudioProcessor(model_size="base")
            self.text_synchronizer = TextSynchronizer()
            self.video_processor = ProVideoProcessor()
            self.script_matcher = ScriptMatcher()
            self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
            logger.info("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def process_reel(self, 
                     audio_file: str,
                     video_file: str,
                     original_script: str,  # NEW: Original script input
                     font_family: str,
                     font_size: int,
                     text_color: str,
                     stroke_color: str,
                     stroke_width: int,
                     position: str,
                     text_case: str,
                     line_spacing: int,
                     letter_spacing: int,
                     vignette_strength: float,
                     words_per_segment: int) -> Tuple[Optional[str], str]:
        """
        Simplified processing function with vignette and script input
        """
        try:
            # Cache management - clean up previous generation
            logger.info("🗑️ Cleaning up previous generation...")
            self.video_processor.cleanup_previous_session()
            
            # Validate inputs
            if not audio_file or not video_file:
                return None, "❌ Please upload both audio and video files"
            
            logger.info("🎨 Starting reel processing...")
            
            # Step 1: Process audio
            status = "🎵 Processing audio...\n"
            
            if not self.audio_processor.validate_audio_file(audio_file):
                return None, status + "❌ Invalid audio file"
            
            status += "✅ Audio validated\n"
            
            # Step 2: Transcription
            status += "🤖 Transcribing audio...\n"
            
            transcription = self.audio_processor.transcribe_with_timestamps(audio_file)
            
            if not transcription['words']:
                return None, status + "❌ Failed to transcribe audio. Please check audio quality."
            
            status += f"✅ Transcribed {len(transcription['words'])} words\n"
            
            # Step 3: Script Matching (NEW FEATURE)
            if original_script and original_script.strip():
                status += "📝 Matching script to audio timing...\n"
                
                # Match provided script with Whisper timing
                matched_words = self.script_matcher.match_script_to_timing(
                    original_script, transcription['words']
                )
                
                # Update transcription with matched words
                transcription['words'] = matched_words
                transcription['text'] = ' '.join([w['text'] for w in matched_words])
                
                # Count matching statistics
                script_matched = sum(1 for w in matched_words if w.get('source') == 'script_matched')
                total_words = len(matched_words)
                
                status += f"✅ Script matching complete: {script_matched}/{total_words} words matched\n"
                logger.info(f"Script matching: {script_matched}/{total_words} words matched")
            else:
                status += "📝 Using Whisper transcription (no script provided)\n"
            
            # Calculate duration: audio + 4 seconds
            audio_duration = transcription['duration']
            total_duration = audio_duration + 4.0
            
            status += f"⏱️ Total duration: {total_duration:.1f}s\n"
            
            # Step 4: Create text segments
            status += "📝 Creating text segments...\n"
            
            self.text_synchronizer.words_per_display = words_per_segment
            text_segments = self.text_synchronizer.create_display_segments(
                transcription['words'], total_duration
            )
            
            if not text_segments:
                return None, status + "❌ Failed to create text segments"
            
            status += f"✅ Created {len(text_segments)} text segments\n"
            
            # Step 5: Process background video with vignette
            status += f"🎬 Processing background video with vignette ({vignette_strength:.1f})...\n"
            
            processed_video_path = self.video_processor.process_background_video(
                video_file, total_duration, vignette_strength
            )
            
            status += "✅ Background video processed with vignette\n"
            
            # Step 6: Create text overlays
            status += "✨ Creating text overlays...\n"
            
            style_config = {
                'font_family': font_family,
                'font_size': font_size,
                'text_color': text_color,
                'stroke_color': stroke_color,
                'stroke_width': stroke_width,
                'position': position,
                'text_case': text_case,
                'line_spacing': line_spacing,
                'letter_spacing': letter_spacing,
            }
            
            text_overlays = self.video_processor.create_text_overlays(text_segments, style_config)
            
            status += f"✅ Created {len(text_overlays)} text overlays\n"
            
            # Step 7: Compose final video
            status += "🎭 Composing final video...\n"
            
            video_with_text = self.video_processor.compose_final_video(processed_video_path, text_overlays)
            
            status += "✅ Video composition complete\n"
            
            # Step 8: Add audio track
            status += "🔊 Adding audio track...\n"
            
            video_with_audio = self.video_processor.add_audio_track(video_with_text, audio_file)
            
            status += "✅ Audio track added\n"
            
            # Step 9: Export final video
            status += "💾 Exporting video...\n"
            
            final_video_path = self.video_processor.export_video(video_with_audio)
            
            # Create final status with script information
            script_info = ""
            if original_script and original_script.strip():
                matched_words = transcription['words']
                script_matched = sum(1 for w in matched_words if w.get('source') == 'script_matched')
                total_words = len(matched_words)
                script_info = f"\n🎯 **Script Accuracy:** {script_matched}/{total_words} words matched from provided script"
            
            final_status = status + f"""
🎉 **Success! Your reel is ready!**

📊 **Video Statistics:**
- Total Duration: {total_duration:.2f} seconds
- Resolution: 1080x1920 (9:16 aspect ratio)
- Text segments: {len(text_segments)} ({words_per_segment} words each)
- Frame rate: 30 FPS
- Font: {font_family} ({font_size}px)
- Text case: {text_case.title()}
- Vignette: {vignette_strength:.1f} strength{script_info}

📝 **Final Text:**
{transcription['text'][:300]}{'...' if len(transcription['text']) > 300 else ''}

🎉 Your reel is ready for social media!
            """
            
            logger.info("🎨 Reel processing completed successfully")
            return final_video_path, final_status
            
        except Exception as e:
            error_message = f"""❌ **Processing Error:** {str(e)}

🔧 **Troubleshooting:**
- Check that both audio and video files are valid
- Ensure your script matches the audio content if provided
- Try with shorter video files if processing fails

📞 **Support:**
- Use common formats: MP4 for video, MP3/WAV for audio
- Script input is optional but improves accuracy when provided
- App automatically manages all temporary files
            """
            logger.error(f"Reel processing failed: {e}")
            return None, error_message

def create_live_preview(font_family, font_size, text_color, stroke_color, stroke_width, position, text_case, line_spacing, letter_spacing, vignette_strength, words_per_segment):
    """Create live preview of text styling with vignette effect"""
    
    # Sample text for preview based on words per segment
    sample_words = ["This", "is", "how", "your", "text", "will", "look", "in", "the", "final", "video"]
    if words_per_segment == 1:
        sample_text = "This"
    elif words_per_segment == 2:
        sample_text = "This is"
    elif words_per_segment == 3:
        sample_text = "This is how"
    elif words_per_segment == 4:
        sample_text = "This is how your"
    else:
        sample_text = "This is how your text"
    
    # Apply text case
    if text_case == 'uppercase':
        display_text = sample_text.upper()
    elif text_case == 'lowercase':
        display_text = sample_text.lower()
    else:
        display_text = sample_text
    
    # Create CSS styles
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
            font-family: '{clean_font}', Arial, sans-serif;
            font-size: {max(12, font_size // 4)}px;
            font-weight: {font_weight};
            color: {text_color};
            text-align: center;
            line-height: {1.0 + (line_spacing / 100)};
            letter-spacing: {letter_spacing / 4}px;
            text-shadow: 
                {stroke_width}px {stroke_width}px 0px {stroke_color},
                -{stroke_width}px {stroke_width}px 0px {stroke_color},
                {stroke_width}px -{stroke_width}px 0px {stroke_color},
                -{stroke_width}px -{stroke_width}px 0px {stroke_color};
            max-width: 90%;
            word-wrap: break-word;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 5;
            position: relative;
        ">{display_text}</div>
    </div>
    """
    
    return preview_html

def create_interface() -> gr.Blocks:
    """Create interface with tabs, dark theme, and script input"""
    
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
        title="Professional Reel Maker with Script Input",
        css=professional_css,
        theme=gr.themes.Base()
    ) as interface:
        
        # Professional header
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 Professional Reel Maker</h1>
            <p>Create engaging vertical videos with AI-powered transcription and professional text overlays</p>
            <div style="margin-top: 15px;">
                <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🎨 Professional Grade</span>
                <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">📱 Live Preview</span>
                <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">🌅 Vignette</span>
                <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; margin: 3px; font-weight: bold;">📝 Script Input</span>
            </div>
        </div>
        """)
        
        # Create tabbed interface
        with gr.Tabs():
            
            # TAB 1: GENERATE - Upload, Process & Results
            with gr.TabItem("🎬 Generate Reel"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>📤 Upload Your Content</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            audio_input = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="🎵 Audio/Voice Recording",
                                show_download_button=False
                            )
                            
                            video_input = gr.Video(
                                sources=["upload"],
                                label="🎬 Background Video",
                                height=200
                            )
                        
                        # NEW: Script Input Section
                        gr.HTML('<div class="section-header"><h3>📝 Original Script (Optional)</h3></div>')
                        
                        with gr.Group(elem_classes="script-container"):
                            original_script = gr.Textbox(
                                label="📄 Original Script/Transcript",
                                placeholder="Paste your original script here for improved accuracy...\n\nExample:\nHello everyone, welcome to my channel.\nToday we're going to talk about...",
                                lines=6,
                                value="",
                                info="Optional: Provide the original script to improve text accuracy. The app will match your script with audio timing."
                            )
                        
                        # Professional Process button
                        process_btn = gr.Button(
                            "🎬 Create Professional Reel",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Performance info
                        gr.HTML("""
                        <div class="performance-info">
                            <strong>🚀 Professional Features</strong><br>
                            🎨 Live preview • 📱 9:16 aspect ratio • 🌅 Background vignette<br>
                            🔤 Smart text wrapping • 🎵 AI transcription • 📝 Script matching • ✅ Clean rendering
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>📱 Your Professional Reel</h3></div>')
                        
                        with gr.Group(elem_classes="controls-container"):
                            video_output = gr.Video(
                                label="Professional Quality Reel",
                                height=400,
                                show_download_button=True
                            )
                            
                            status_output = gr.Textbox(
                                label="Real-time Processing Status",
                                lines=12,
                                interactive=False
                            )
            
            # TAB 2: STYLE & PREVIEW - All Controls with Live Preview
            with gr.TabItem("🎨 Style & Preview"):
                gr.HTML('<div class="section-header"><h3>🎨 Text Styling with Live Preview</h3></div>')
                
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
                                    label="Font Size"
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
                                maximum=10,
                                value=3,
                                step=1,
                                label="Outline Width"
                            )
                            
                            position = gr.Dropdown(
                                choices=["top", "center", "bottom"],
                                value="center",
                                label="Text Position"
                            )
                            
                            with gr.Row():
                                line_spacing = gr.Slider(
                                    minimum=0,
                                    maximum=50,
                                    value=10,
                                    step=2,
                                    label="Line Spacing"
                                )
                                
                                letter_spacing = gr.Slider(
                                    minimum=0,
                                    maximum=20,
                                    value=0,
                                    step=1,
                                    label="Letter Spacing"
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
                                label="📝 Words per text segment",
                                info="How many words appear together"
                            )
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-header"><h3>👁️ Live Preview</h3></div>')
                        
                        with gr.Group(elem_classes="preview-container"):
                            live_preview = gr.HTML(
                                create_live_preview(
                                    "Arial Bold", 60, "#FFFFFF", "#000000", 3, 
                                    "center", "regular", 10, 0, 0.3, 2
                                ),
                                label="Live Text Preview"
                            )
                            
                            gr.HTML("""
                            <div style="text-align: center; margin-top: 20px; color: #a0aec0; font-size: 14px;">
                                <strong>🎯 Preview Features:</strong><br>
                                • Real-time font changes<br>
                                • Accurate 9:16 aspect ratio<br>
                                • Live color updates<br>
                                • Position preview<br>
                                • Vignette effect preview<br>
                                • Words per segment preview
                            </div>
                            """)
        
        # Update live preview when any styling option changes
        style_inputs = [
            font_family, font_size, text_color, stroke_color, stroke_width,
            position, text_case, line_spacing, letter_spacing, vignette_strength, words_per_segment
        ]
        
        for input_component in style_inputs:
            input_component.change(
                fn=create_live_preview,
                inputs=style_inputs,
                outputs=[live_preview]
            )
        
        # Connect processing function with script input
        process_btn.click(
            fn=app.process_reel,
            inputs=[
                audio_input,
                video_input,
                original_script,  # NEW: Include script input
                font_family,
                font_size,
                text_color,
                stroke_color,
                stroke_width,
                position,
                text_case,
                line_spacing,
                letter_spacing,
                vignette_strength,
                words_per_segment
            ],
            outputs=[video_output, status_output]
        )
    
    return interface

def main():
    """Main entry point"""
    logger.info("🎬 Starting Professional Reel Maker with Script Input")
    
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