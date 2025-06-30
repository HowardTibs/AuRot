"""
Smart text synchronization for optimal display timing
Creates readable text segments that match audio pacing
NO FADE EFFECTS - Text appears instantly and stays until next text appears
ENHANCED with Reddit integration - supports offset timing for dual audio system
UPDATED: Added dead air support - last text remains visible during ending silence
FIXED: Proper segment extension and timing calculations for 3.5s dead air
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TextSynchronizer:
    """Handles intelligent text segmentation and timing with INSTANT display + Reddit integration + Dead air"""
    
    def __init__(self, 
                 words_per_display: int = 2,
                 min_display_time: float = 0.8,
                 max_display_time: float = 4.0):
        """
        Initialize text synchronizer - NO FADE EFFECTS + Reddit integration + Dead air support
        
        Args:
            words_per_display: Target number of words per text display
            min_display_time: Minimum time to show each text segment
            max_display_time: Maximum time to show each text segment
        """
        self.words_per_display = words_per_display
        self.min_display_time = min_display_time
        self.max_display_time = max_display_time
        # REMOVED: No fade duration - instant display only
    
    def create_display_segments(self, words: List[Dict], total_video_duration: float = None) -> List[Dict]:
        """
        Create optimized text segments with INSTANT display timing
        
        Args:
            words: List of word dictionaries with 'text', 'start', 'end' keys
            total_video_duration: Total video duration to extend last segment
            
        Returns:
            List of text segments with instant display timing
        """
        if not words:
            logger.warning("No words provided for segmentation")
            return []
        
        segments = []
        current_words = []
        
        logger.info(f"Creating segments from {len(words)} words with INSTANT display")
        
        for i, word in enumerate(words):
            current_words.append(word)
            
            # Decide when to create a segment
            should_create_segment = self._should_create_segment(
                current_words, word, words[i+1:] if i+1 < len(words) else []
            )
            
            if should_create_segment or i == len(words) - 1:  # Last word
                segment = self._create_segment_from_words(current_words)
                if segment:
                    segments.append(segment)
                current_words = []
        
        # SIMPLIFIED: Basic segment timing optimization - no overlaps
        optimized_segments = self._optimize_segment_timing_instant(segments)
        
        # Extend last segment to cover full video duration
        if optimized_segments and total_video_duration:
            last_segment = optimized_segments[-1]
            if last_segment['end'] < total_video_duration:
                logger.info(f"Extending last segment from {last_segment['end']:.2f}s to {total_video_duration:.2f}s")
                last_segment['end'] = total_video_duration
                last_segment['duration'] = total_video_duration - last_segment['start']
                
                # Log the extension
                extension_time = total_video_duration - last_segment['start'] - last_segment['natural_duration']
                logger.info(f"Last text segment extended by {extension_time:.2f}s to cover buffer")
        
        logger.info(f"Created {len(optimized_segments)} display segments with INSTANT display")
        return optimized_segments
    
    def create_display_segments_with_offset(self, words: List[Dict], total_video_duration: float, offset_duration: float) -> List[Dict]:
        """
        Create text segments with time offset for Reddit integration
        
        Args:
            words: List of word dictionaries with 'text', 'start', 'end' keys  
            total_video_duration: Total video duration to extend last segment
            offset_duration: Time offset to add to all segments (Reddit display + delay time)
            
        Returns:
            List of text segments with offset timing for Reddit integration
        """
        if not words:
            logger.warning("No words provided for segmentation with offset")
            return []
        
        logger.info(f"Creating segments with {offset_duration:.2f}s offset for Reddit integration")
        
        # First create normal segments
        segments = []
        current_words = []
        
        for i, word in enumerate(words):
            current_words.append(word)
            
            # Decide when to create a segment
            should_create_segment = self._should_create_segment(
                current_words, word, words[i+1:] if i+1 < len(words) else []
            )
            
            if should_create_segment or i == len(words) - 1:  # Last word
                segment = self._create_segment_from_words(current_words)
                if segment:
                    segments.append(segment)
                current_words = []
        
        # Apply offset to all segments
        offset_segments = []
        for segment in segments:
            offset_segment = segment.copy()
            offset_segment['start'] = segment['start'] + offset_duration
            offset_segment['end'] = segment['end'] + offset_duration
            # Keep same duration
            offset_segments.append(offset_segment)
        
        # Optimize timing with offset
        optimized_segments = self._optimize_segment_timing_instant(offset_segments)
        
        # Extend last segment to cover full video duration
        if optimized_segments and total_video_duration:
            last_segment = optimized_segments[-1]
            if last_segment['end'] < total_video_duration:
                logger.info(f"Extending last offset segment from {last_segment['end']:.2f}s to {total_video_duration:.2f}s")
                last_segment['end'] = total_video_duration
                last_segment['duration'] = total_video_duration - last_segment['start']
        
        logger.info(f"Created {len(optimized_segments)} offset display segments (starting at {offset_duration:.2f}s)")
        return optimized_segments
    
    def create_display_segments_with_offset_and_extended_end(self, words: List[Dict], total_video_duration: float, 
                                                           offset_duration: float, ending_silence_duration: float) -> List[Dict]:
        """
        FIXED: Create text segments with time offset for Reddit integration and extended last segment for dead air
        
        Args:
            words: List of word dictionaries with 'text', 'start', 'end' keys  
            total_video_duration: Total video duration to extend last segment
            offset_duration: Time offset to add to all segments (Reddit display + delay time)
            ending_silence_duration: Duration of ending silence for dead air
            
        Returns:
            List of text segments with offset timing and extended last segment for dead air
        """
        if not words:
            logger.warning("No words provided for segmentation with offset and extended end")
            return []
        
        logger.info(f"Creating segments with {offset_duration:.2f}s offset and {ending_silence_duration:.2f}s dead air extension")
        
        # First create normal segments
        segments = []
        current_words = []
        
        for i, word in enumerate(words):
            current_words.append(word)
            
            # Decide when to create a segment
            should_create_segment = self._should_create_segment(
                current_words, word, words[i+1:] if i+1 < len(words) else []
            )
            
            if should_create_segment or i == len(words) - 1:  # Last word
                segment = self._create_segment_from_words(current_words)
                if segment:
                    segments.append(segment)
                current_words = []
        
        # Apply offset to all segments
        offset_segments = []
        for segment in segments:
            offset_segment = segment.copy()
            offset_segment['start'] = segment['start'] + offset_duration
            offset_segment['end'] = segment['end'] + offset_duration
            # Keep same duration
            offset_segments.append(offset_segment)
        
        # Optimize timing with offset
        optimized_segments = self._optimize_segment_timing_instant(offset_segments)
        
        # FIXED: ALWAYS extend last segment to cover dead air period
        if optimized_segments and total_video_duration:
            last_segment = optimized_segments[-1]
            
            # Calculate the natural end time (without dead air extension)
            audio_end_time = total_video_duration - ending_silence_duration
            
            # ALWAYS extend the last segment to the total duration (including dead air)
            original_end = last_segment['end']
            last_segment['end'] = total_video_duration
            last_segment['duration'] = total_video_duration - last_segment['start']
            
            # Mark this segment as having dead air extension
            last_segment['has_dead_air_extension'] = True
            last_segment['natural_end'] = original_end
            last_segment['dead_air_start'] = audio_end_time
            last_segment['dead_air_duration'] = ending_silence_duration
            
            logger.info(f"🔥 FIXED: Extended last segment to cover ENTIRE dead air period:")
            logger.info(f"  • Segment text: '{last_segment['text'][:50]}{'...' if len(last_segment['text']) > 50 else ''}'")
            logger.info(f"  • Original end: {original_end:.2f}s")
            logger.info(f"  • Audio ends at: {audio_end_time:.2f}s") 
            logger.info(f"  • NEW segment end: {total_video_duration:.2f}s")
            logger.info(f"  • Dead air duration: {ending_silence_duration:.2f}s")
            logger.info(f"  • Text will stay visible for {total_video_duration - original_end:.2f}s extra")
        else:
            logger.warning("No segments to extend for dead air")
        
        logger.info(f"Created {len(optimized_segments)} segments with offset and dead air extension")
        return optimized_segments
    
    def _should_create_segment(self, current_words: List[Dict], current_word: Dict, remaining_words: List[Dict]) -> bool:
        """
        Determine if we should create a segment at this point
        
        Args:
            current_words: Words accumulated so far
            current_word: Current word being processed
            remaining_words: Words that come after current word
            
        Returns:
            True if segment should be created
        """
        # Check word count threshold
        if len(current_words) >= self.words_per_display:
            return True
        
        # Check for natural breaks (punctuation)
        if self._has_natural_break(current_word):
            return True
        
        # Check for significant pause
        if remaining_words and self._has_significant_pause(current_word, remaining_words[0]):
            return True
        
        # Check if current segment is getting too long
        if len(current_words) > 1:
            duration = current_words[-1]['end'] - current_words[0]['start']
            if duration > self.max_display_time:
                return True
        
        return False
    
    def _has_natural_break(self, word: Dict) -> bool:
        """Check if word contains natural break points"""
        text = word.get('text', '')
        break_chars = '.!?,:;'
        return any(char in text for char in break_chars)
    
    def _has_significant_pause(self, current_word: Dict, next_word: Dict) -> bool:
        """Check if there's a significant pause between words"""
        pause_duration = next_word.get('start', 0) - current_word.get('end', 0)
        return pause_duration > 0.5  # 500ms threshold
    
    def _create_segment_from_words(self, words: List[Dict]) -> Dict:
        """
        Create a display segment from a group of words
        
        Args:
            words: List of word dictionaries
            
        Returns:
            Segment dictionary with display information
        """
        if not words:
            return None
        
        # Combine text
        segment_text = ' '.join([w.get('text', '').strip() for w in words])
        
        # Calculate timing
        start_time = words[0].get('start', 0)
        end_time = words[-1].get('end', 0)
        natural_duration = end_time - start_time
        
        # Ensure minimum display time
        display_duration = max(natural_duration, self.min_display_time)
        display_duration = min(display_duration, self.max_display_time)
        
        # Calculate confidence (average of word confidences)
        confidences = [w.get('confidence', 0.9) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
        
        segment = {
            'text': segment_text,
            'start': start_time,
            'end': start_time + display_duration,
            'duration': display_duration,
            'word_count': len(words),
            'confidence': avg_confidence,
            'natural_duration': natural_duration,
            'has_dead_air_extension': False  # Default - will be set for last segment if needed
        }
        
        return segment
    
    def _optimize_segment_timing_instant(self, segments: List[Dict]) -> List[Dict]:
        """
        SIMPLIFIED: Instant display timing optimization - no fade coordination needed
        
        Args:
            segments: List of segments to optimize
            
        Returns:
            Optimized segments list for instant display
        """
        if not segments:
            return []
        
        optimized = []
        
        for i, segment in enumerate(segments):
            optimized_segment = segment.copy()
            
            # SIMPLIFIED: Just prevent overlaps - no fade timing needed
            if i > 0:
                prev_segment = optimized[-1]
                gap = segment['start'] - prev_segment['end']
                
                if gap < 0:  # Overlap - fix it
                    # Trim previous segment to prevent overlap
                    overlap_amount = abs(gap)
                    new_prev_end = prev_segment['end'] - overlap_amount - 0.05  # 50ms gap
                    optimized[-1]['end'] = new_prev_end
                    optimized[-1]['duration'] = new_prev_end - prev_segment['start']
                    
                    logger.debug(f"Trimmed previous segment to prevent overlap: {overlap_amount:.2f}s")
            
            # Ensure segment doesn't overlap with next (unless it's the last segment with dead air)
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                
                if optimized_segment['end'] > next_segment['start']:
                    # Trim segment to prevent overlap
                    optimized_segment['end'] = next_segment['start'] - 0.05  # 50ms gap
                    optimized_segment['duration'] = optimized_segment['end'] - optimized_segment['start']
                    
                    # Ensure we don't make the segment too short
                    if optimized_segment['duration'] < self.min_display_time:
                        optimized_segment['duration'] = self.min_display_time
                        optimized_segment['end'] = optimized_segment['start'] + self.min_display_time
            
            optimized.append(optimized_segment)
        
        return optimized
    
    def get_segment_stats(self, segments: List[Dict]) -> Dict:
        """
        Get statistics about the created segments including dead air info
        
        Args:
            segments: List of segments
            
        Returns:
            Statistics dictionary
        """
        if not segments:
            return {'total_segments': 0}
        
        total_duration = sum(seg['duration'] for seg in segments)
        avg_duration = total_duration / len(segments)
        avg_words = sum(seg['word_count'] for seg in segments) / len(segments)
        avg_confidence = sum(seg['confidence'] for seg in segments) / len(segments)
        
        # Check for dead air extension
        dead_air_info = {}
        last_segment = segments[-1] if segments else None
        if last_segment and last_segment.get('has_dead_air_extension', False):
            dead_air_info = {
                'has_dead_air': True,
                'dead_air_duration': last_segment.get('dead_air_duration', 0),
                'dead_air_start': last_segment.get('dead_air_start', 0),
                'natural_end': last_segment.get('natural_end', 0)
            }
        else:
            dead_air_info = {'has_dead_air': False}
        
        return {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'average_words_per_segment': avg_words,
            'average_confidence': avg_confidence,
            'shortest_segment': min(seg['duration'] for seg in segments),
            'longest_segment': max(seg['duration'] for seg in segments),
            'display_mode': 'instant',  # NO FADE EFFECTS
            'reddit_integration': 'supported',  # Reddit integration support
            'dead_air_info': dead_air_info  # NEW: Dead air information
        }
    
    def calculate_reddit_timing(self, title_duration: float, delay_duration: float = 1.0) -> Dict:
        """
        Calculate timing information for Reddit integration
        
        Args:
            title_duration: Duration of title audio
            delay_duration: Delay between title and main audio
            
        Returns:
            Dictionary with timing information for Reddit integration
        """
        reddit_display_duration = title_duration + delay_duration
        
        timing_info = {
            'title_duration': title_duration,
            'delay_duration': delay_duration,
            'reddit_display_duration': reddit_display_duration,
            'text_start_time': reddit_display_duration,
            'reddit_phases': {
                'title_phase': {'start': 0.0, 'end': title_duration},
                'delay_phase': {'start': title_duration, 'end': reddit_display_duration},
                'text_phase': {'start': reddit_display_duration, 'end': None}  # Will be set later
            }
        }
        
        logger.info(f"Reddit timing calculated: {reddit_display_duration:.2f}s display, text starts at {reddit_display_duration:.2f}s")
        return timing_info
    
    def calculate_dead_air_timing(self, audio_duration: float, ending_silence: float) -> Dict:
        """
        Calculate timing information for dead air phase
        
        Args:
            audio_duration: Duration of actual audio content
            ending_silence: Duration of ending silence (dead air)
            
        Returns:
            Dictionary with dead air timing information
        """
        total_duration = audio_duration + ending_silence
        dead_air_start = audio_duration
        
        timing_info = {
            'audio_duration': audio_duration,
            'ending_silence': ending_silence,
            'total_duration': total_duration,
            'dead_air_start': dead_air_start,
            'dead_air_end': total_duration,
            'phases': {
                'audio_phase': {'start': 0.0, 'end': audio_duration},
                'dead_air_phase': {'start': dead_air_start, 'end': total_duration}
            }
        }
        
        logger.info(f"Dead air timing calculated: {ending_silence:.2f}s dead air starts at {dead_air_start:.2f}s")
        return timing_info
    
    def validate_dead_air_extension(self, segments: List[Dict], total_duration: float, dead_air_start: float) -> bool:
        """
        NEW: Validate that dead air extension is working correctly
        
        Args:
            segments: List of text segments
            total_duration: Total video duration
            dead_air_start: When dead air begins
            
        Returns:
            True if dead air extension is properly configured
        """
        if not segments:
            logger.warning("No segments to validate for dead air")
            return False
        
        last_segment = segments[-1]
        
        # Check if last segment has dead air extension
        if not last_segment.get('has_dead_air_extension', False):
            logger.warning("Last segment does not have dead air extension")
            return False
        
        # Check if last segment extends to total duration
        if abs(last_segment['end'] - total_duration) > 0.1:
            logger.warning(f"Last segment end ({last_segment['end']:.2f}s) doesn't match total duration ({total_duration:.2f}s)")
            return False
        
        # Check if dead air start time is correct
        segment_dead_air_start = last_segment.get('dead_air_start', 0)
        if abs(segment_dead_air_start - dead_air_start) > 0.1:
            logger.warning(f"Dead air start mismatch: segment={segment_dead_air_start:.2f}s, expected={dead_air_start:.2f}s")
            return False
        
        logger.info("✅ Dead air extension validation passed")
        return True