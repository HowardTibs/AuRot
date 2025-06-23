"""
Smart text synchronization for optimal display timing
Creates readable text segments that match audio pacing
FIXED: Last segment now extends to cover full video duration
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TextSynchronizer:
    """Handles intelligent text segmentation and timing"""
    
    def __init__(self, 
                 words_per_display: int = 2,
                 min_display_time: float = 0.8,
                 max_display_time: float = 4.0):
        """
        Initialize text synchronizer
        
        Args:
            words_per_display: Target number of words per text display
            min_display_time: Minimum time to show each text segment
            max_display_time: Maximum time to show each text segment
        """
        self.words_per_display = words_per_display
        self.min_display_time = min_display_time
        self.max_display_time = max_display_time
    
    def create_display_segments(self, words: List[Dict], total_video_duration: float = None) -> List[Dict]:
        """
        Create optimized text segments from word timestamps
        
        Args:
            words: List of word dictionaries with 'text', 'start', 'end' keys
            total_video_duration: Total video duration to extend last segment
            
        Returns:
            List of text segments with timing and display information
        """
        if not words:
            logger.warning("No words provided for segmentation")
            return []
        
        segments = []
        current_words = []
        
        logger.info(f"Creating segments from {len(words)} words")
        
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
        
        # Post-process segments for optimal timing
        optimized_segments = self._optimize_segment_timing(segments)
        
        # FIXED: Extend last segment to cover full video duration
        if optimized_segments and total_video_duration:
            last_segment = optimized_segments[-1]
            if last_segment['end'] < total_video_duration:
                logger.info(f"Extending last segment from {last_segment['end']:.2f}s to {total_video_duration:.2f}s")
                last_segment['end'] = total_video_duration
                last_segment['duration'] = total_video_duration - last_segment['start']
                
                # Log the extension
                extension_time = total_video_duration - last_segment['start'] - last_segment['natural_duration']
                logger.info(f"Last text segment extended by {extension_time:.2f}s to cover 4-second buffer")
        
        logger.info(f"Created {len(optimized_segments)} display segments")
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
            'natural_duration': natural_duration
        }
        
        return segment
    
    def _optimize_segment_timing(self, segments: List[Dict]) -> List[Dict]:
        """
        Optimize segment timing to prevent overlaps and ensure smooth transitions
        
        Args:
            segments: List of segments to optimize
            
        Returns:
            Optimized segments list
        """
        if not segments:
            return []
        
        optimized = []
        
        for i, segment in enumerate(segments):
            optimized_segment = segment.copy()
            
            # Ensure no overlap with previous segment
            if i > 0:
                prev_segment = optimized[-1]
                gap = segment['start'] - prev_segment['end']
                
                if gap < 0.1:  # Minimum 100ms gap
                    # Adjust start time to prevent overlap
                    new_start = prev_segment['end'] + 0.1
                    duration_adjustment = new_start - segment['start']
                    
                    optimized_segment['start'] = new_start
                    optimized_segment['end'] = new_start + segment['duration']
                    
                    logger.debug(f"Adjusted segment {i} timing to prevent overlap")
            
            # Ensure segment doesn't exceed bounds (except for last segment which will be extended later)
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                if optimized_segment['end'] > next_segment['start']:
                    # Trim segment to not overlap with next
                    optimized_segment['end'] = next_segment['start'] - 0.1
                    optimized_segment['duration'] = optimized_segment['end'] - optimized_segment['start']
            
            optimized.append(optimized_segment)
        
        return optimized
    
    def get_segment_stats(self, segments: List[Dict]) -> Dict:
        """
        Get statistics about the created segments
        
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
        
        return {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'average_words_per_segment': avg_words,
            'average_confidence': avg_confidence,
            'shortest_segment': min(seg['duration'] for seg in segments),
            'longest_segment': max(seg['duration'] for seg in segments)
        }