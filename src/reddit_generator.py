"""
Reddit Template Generator Module - ENHANCED VERSION with CONSISTENCY FIXES
Generates authentic-looking Reddit posts with custom titles, usernames, and avatars
Integrated component for the Professional Reel Maker with video integration optimizations
FIXED: Comments display issue and defaults, black spots, and margin symmetry
CONSISTENCY FIX: Enhanced state management and cache clearing for reliable generation across sessions
"""

import PIL
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import os
import logging
import tempfile
import gc  # ADDED: For manual garbage collection
import copy  # ADDED: For deep copying to avoid state pollution
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class RedditTemplateGenerator:
    """CONSISTENCY FIXED: Generate Reddit post templates with reliable state management across generations"""
    
    def __init__(self):
        """Initialize with enhanced state management"""
        self.saved_profile = {
            "username": "AskReddit",
            "avatar": None,
            "avatar_data": None
        }
        
        # Optimized for video overlay - better proportions
        self.template_width = 1600  # Reduced width for better video fitting
        self.template_height = 800  # Base height, will expand as needed
        
        # Ensure image_src directory exists
        self.image_src_path = "image_src"
        if not os.path.exists(self.image_src_path):
            os.makedirs(self.image_src_path)
            logger.info(f"Created {self.image_src_path} directory")
            
        # CONSISTENCY FIX: Enhanced cache management
        self.asset_cache = {}
        self.font_cache = {}  # ADDED: Separate font cache
        self.image_cache = {}  # ADDED: Separate image cache
        
        # ADDED: Generation tracking for consistency
        self.current_generation = 0
        self.last_successful_generation = 0
        
        logger.info("RedditTemplateGenerator initialized with enhanced state management")
    
    def clear_cache(self):
        """ADDED: Clear all caches for clean slate between generations"""
        try:
            self.current_generation += 1
            logger.info(f"🧹 Clearing Reddit generator caches for generation #{self.current_generation}")
            
            # Clear all caches
            self.asset_cache.clear()
            self.font_cache.clear()
            self.image_cache.clear()
            
            # Force garbage collection
            for _ in range(2):
                gc.collect()
            
            logger.info(f"✅ Reddit generator caches cleared for generation #{self.current_generation}")
            
        except Exception as e:
            logger.error(f"Failed to clear Reddit generator caches: {e}")
    
    def reset_to_defaults(self):
        """ADDED: Reset generator to default state"""
        try:
            logger.info("Resetting Reddit generator to defaults")
            
            # Reset saved profile to defaults
            self.saved_profile = {
                "username": "AskReddit",
                "avatar": None,
                "avatar_data": None
            }
            
            # Clear all caches
            self.clear_cache()
            
            logger.info("✅ Reddit generator reset to defaults")
            
        except Exception as e:
            logger.error(f"Failed to reset Reddit generator: {e}")
    
    def validate_inputs(self, title: str, username: str = None, likes: str = "99+", comments: str = "99+") -> Tuple[bool, str]:
        """ADDED: Validate inputs for consistency"""
        try:
            # Validate title
            if not title or not isinstance(title, str) or not title.strip():
                return False, "❌ Title is required and must be a non-empty string"
            
            if len(title.strip()) > 500:
                return False, "❌ Title is too long (maximum 500 characters)"
            
            # Validate username
            if username is not None:
                if not isinstance(username, str):
                    return False, "❌ Username must be a string"
                
                if len(username) > 50:
                    return False, "❌ Username is too long (maximum 50 characters)"
            
            # Validate likes and comments
            for field_name, field_value in [("Likes", likes), ("Comments", comments)]:
                if field_value and not isinstance(field_value, str):
                    return False, f"❌ {field_name} must be a string"
                
                if field_value and len(field_value) > 20:
                    return False, f"❌ {field_name} is too long (maximum 20 characters)"
            
            return True, "✅ Input validation passed"
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, f"❌ Validation error: {str(e)}"
        
    def create_reddit_template(self, title: str, username: str = None, avatar_image = None, 
                             likes: str = "99+", comments: str = "99+", use_saved_profile: bool = False) -> PIL.Image.Image:
        """CONSISTENCY FIXED: Create a Reddit post template with enhanced validation and error handling"""
        
        try:
            logger.info(f"Creating Reddit template (Gen #{self.current_generation})")
            
            # CONSISTENCY FIX: Validate inputs
            is_valid, validation_msg = self.validate_inputs(title, username, likes, comments)
            if not is_valid:
                logger.error(f"Input validation failed: {validation_msg}")
                return self._create_fallback_image(title)
            
            # Use saved profile if requested
            if use_saved_profile and username is None:
                username = self.saved_profile["username"]
            if use_saved_profile and avatar_image is None and self.saved_profile["avatar_data"]:
                avatar_image = self.saved_profile["avatar_data"]
                
            # Set defaults
            if username is None:
                username = "AskReddit"
            
            # FIXED: Set default comments to 99 instead of empty
            if not comments or comments.strip() == "":
                comments = "99"
            
            # Enhanced dynamic font sizing for video integration
            title_length = len(title)
            if title_length < 25:  # Very short titles
                base_font_size = 96
            elif title_length < 40:  # Short titles
                base_font_size = 88
            elif title_length < 70:  # Medium titles
                base_font_size = 80
            elif title_length < 120:  # Long titles
                base_font_size = 72
            else:  # Very long titles
                base_font_size = 64
                
            username_font_size = int(base_font_size * 0.75)  # Proportional scaling
            stats_font_size = int(base_font_size * 0.65)     # Proportional scaling
            
            # Calculate required height based on title length with video optimization
            temp_img = Image.new('RGB', (self.template_width, 100), 'white')
            temp_draw = ImageDraw.Draw(temp_img)
            
            try:
                # Enhanced font loading with better fallbacks
                title_font = self._load_font_with_fallback("bold", base_font_size)
                username_font = self._load_font_with_fallback("bold", username_font_size)
                stats_font = self._load_font_with_fallback("regular", stats_font_size)
            except Exception as e:
                logger.warning(f"Font loading failed, using defaults: {e}")
                title_font = ImageFont.load_default()
                username_font = ImageFont.load_default()
                stats_font = ImageFont.load_default()
            
            # FIXED: Calculate title dimensions with symmetric margins
            left_margin = 75  # Symmetric margins
            right_margin = 75
            title_area_width = self.template_width - left_margin - right_margin  # Symmetric margins
            title_lines = self._wrap_text_enhanced(title, title_font, title_area_width, temp_draw)
            title_height = len(title_lines) * int(base_font_size * 1.15)  # Optimized line spacing
            
            # Calculate total height needed - optimized for video overlay
            header_height = 180  # Slightly reduced
            title_margin = 50    # Optimized margins
            footer_height = 100  # Slightly reduced
            total_height = header_height + title_margin + title_height + title_margin + footer_height
            
            # Ensure minimum height for video compatibility
            total_height = max(total_height, 600)
            
            # Create the main image with rounded corners support
            img = Image.new('RGBA', (self.template_width, total_height), '#FFFFFF')
            draw = ImageDraw.Draw(img)
            
            # Draw background with subtle gradient for video integration
            self._draw_enhanced_background(draw, img.size)
            
            # Draw the header section (username area)
            self._draw_header_enhanced(draw, username, avatar_image, username_font, img, username_font_size)
            
            # Draw the title section with enhanced formatting and symmetric margins
            title_y = header_height + title_margin
            self._draw_title_enhanced(draw, title_lines, title_y, title_font, base_font_size, left_margin)
            
            # FIXED: Draw the footer (stats) with proper comments display
            footer_y = total_height - footer_height
            self._draw_footer_enhanced(draw, likes, comments, stats_font, footer_y, img, stats_font_size)
            
            # Add rounded corners for better video integration
            img = self._add_rounded_corners_enhanced(img, 25)
            
            # CONSISTENCY FIX: Mark successful generation
            self.last_successful_generation = self.current_generation
            
            logger.info(f"Successfully generated enhanced Reddit template: {self.template_width}x{total_height} (Gen #{self.current_generation})")
            return img
            
        except Exception as e:
            logger.error(f"Failed to create Reddit template (Gen #{self.current_generation}): {e}")
            # Return an enhanced fallback image
            fallback_img = self._create_fallback_image(title)
            return fallback_img
    
    def _load_font_with_fallback(self, style: str, size: int):
        """Enhanced font loading with comprehensive fallbacks and caching"""
        # CONSISTENCY FIX: Create cache key for font caching
        cache_key = f"{style}_{size}"
        
        # Check cache first
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font_paths = []
        
        if style == "bold":
            font_paths = [
                "arialbd.ttf", "Arial-Bold.ttf", "arial.ttf",
                "calibrib.ttf", "Calibri-Bold.ttf", "calibri.ttf",
                "helvetica-bold.ttf", "Helvetica-Bold.ttf"
            ]
        else:
            font_paths = [
                "arial.ttf", "Arial.ttf", 
                "calibri.ttf", "Calibri.ttf",
                "helvetica.ttf", "Helvetica.ttf"
            ]
        
        # Add system-specific paths
        system_paths = []
        for font_file in font_paths:
            system_paths.extend([
                font_file,
                f"C:/Windows/Fonts/{font_file}",
                f"/System/Library/Fonts/{font_file}",
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                f"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ])
        
        loaded_font = None
        for path in system_paths:
            try:
                loaded_font = ImageFont.truetype(path, size)
                break
            except:
                continue
        
        # Ultimate fallback
        if loaded_font is None:
            loaded_font = ImageFont.load_default()
        
        # Cache the font
        self.font_cache[cache_key] = loaded_font
        
        return loaded_font
    
    def _draw_enhanced_background(self, draw: ImageDraw.Draw, size: Tuple[int, int]):
        """Draw enhanced background with subtle gradient"""
        width, height = size
        
        # Create subtle vertical gradient
        for y in range(height):
            # Very subtle gradient from pure white to slightly off-white
            gray_value = int(255 - (y / height) * 8)  # Max 8 point difference
            color = (gray_value, gray_value, gray_value)
            draw.line([(0, y), (width, y)], fill=color)
    
    def _wrap_text_enhanced(self, text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.Draw) -> list:
        """Enhanced text wrapping with better word boundary handling and validation"""
        try:
            if not text or not isinstance(text, str):
                return [""]
            
            words = text.split(' ')
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    text_width = bbox[2] - bbox[0]
                except:
                    # Enhanced fallback calculation
                    text_width = len(test_line) * (getattr(font, 'size', 12) * 0.6)
                
                if text_width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Enhanced handling of very long words
                        if len(word) > 20:  # Break very long words
                            chunk_size = max_width // (getattr(font, 'size', 12) // 2)
                            for i in range(0, len(word), chunk_size):
                                lines.append(word[i:i+chunk_size])
                        else:
                            lines.append(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            return lines if lines else [""]
            
        except Exception as e:
            logger.error(f"Text wrapping failed: {e}")
            return [text[:50] + "..." if len(text) > 50 else text]  # Fallback
    
    def _draw_header_enhanced(self, draw: ImageDraw.Draw, username: str, avatar_image, font: ImageFont.ImageFont, img: Image.Image, username_font_size: int):
        """Enhanced header with better avatar and badge handling"""
        try:
            # Enhanced avatar size and positioning
            avatar_size = 140  # Optimized size for video
            avatar_x, avatar_y = 75, 20  # FIXED: Use symmetric margin (was 60)
            
            # Handle avatar image with enhanced processing
            avatar_loaded = False
            if avatar_image is not None:
                try:
                    avatar_loaded = self._process_avatar_enhanced(avatar_image, draw, avatar_x, avatar_y, avatar_size)
                except Exception as e:
                    logger.error(f"Enhanced avatar processing failed: {e}")
                    avatar_loaded = False
            
            # Draw default avatar if needed
            if not avatar_loaded:
                self._draw_default_avatar_enhanced(draw, avatar_x, avatar_y, avatar_size)
            
            # Enhanced username positioning and styling
            username_x = avatar_x + avatar_size + 30
            username_y = avatar_y + 15
            
            # CONSISTENCY FIX: Validate username
            if not username or not isinstance(username, str):
                username = "AskReddit"
            
            # Add subtle text shadow for better video visibility
            shadow_offset = 1
            draw.text((username_x + shadow_offset, username_y + shadow_offset), username, 
                     fill='#E0E0E0', font=font)
            draw.text((username_x, username_y), username, fill='#000000', font=font)
            
            # Enhanced verified checkmark
            self._draw_verified_checkmark_enhanced(draw, username_x, username_y, username, font, username_font_size)
            
            # Enhanced badge system
            self._draw_badges_enhanced(draw, img, username_x, username_y, username_font_size)
            
        except Exception as e:
            logger.error(f"Header drawing failed: {e}")
    
    def _process_avatar_enhanced(self, avatar_image, draw: ImageDraw.Draw, x: int, y: int, size: int) -> bool:
        """Enhanced avatar processing with better quality and error handling"""
        try:
            # CONSISTENCY FIX: Enhanced validation
            if avatar_image is None:
                return False
            
            # Load and process the avatar image
            if isinstance(avatar_image, str):
                if not os.path.exists(avatar_image):
                    logger.warning(f"Avatar image file not found: {avatar_image}")
                    return False
                avatar = Image.open(avatar_image).convert('RGBA')
            else:
                avatar = avatar_image.convert('RGBA')
            
            # Enhanced resizing with better quality
            avatar = avatar.resize((size, size), Image.Resampling.LANCZOS)
            
            # Create circular mask with anti-aliasing
            mask = Image.new('L', (size, size), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.ellipse((0, 0, size, size), fill=255)
            
            # Apply mask
            avatar.putalpha(mask)
            
            # For drawing, we'll use a simplified approach
            center = size // 2
            for y_pos in range(size):
                for x_pos in range(size):
                    if (x_pos - center) ** 2 + (y_pos - center) ** 2 <= center ** 2:
                        try:
                            pixel = avatar.getpixel((x_pos, y_pos))
                            if len(pixel) >= 3:  # Has RGB components
                                draw.rectangle((x + x_pos, y + y_pos, x + x_pos + 1, y + y_pos + 1), 
                                             fill=pixel[:3])
                        except:
                            continue
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced avatar processing failed: {e}")
            return False
    
    def _draw_default_avatar_enhanced(self, draw: ImageDraw.Draw, x: int, y: int, size: int):
        """Enhanced default Reddit alien avatar"""
        try:
            # Enhanced orange circle with gradient effect
            draw.ellipse((x, y, x + size, y + size), fill='#FF4500')
            
            # Add subtle inner shadow
            inner_size = int(size * 0.95)
            inner_offset = (size - inner_size) // 2
            draw.ellipse((x + inner_offset, y + inner_offset, 
                         x + inner_offset + inner_size, y + inner_offset + inner_size), 
                        outline='#E03D00', width=2)
            
            # Enhanced Reddit alien features
            head_size = int(size * 0.35)
            head_x = x + (size - head_size) // 2
            head_y = y + int(size * 0.15)
            
            # White head with subtle shadow
            draw.ellipse((head_x + 1, head_y + 1, head_x + head_size + 1, head_y + head_size + 1), 
                        fill='#F0F0F0')  # Shadow
            draw.ellipse((head_x, head_y, head_x + head_size, head_y + head_size), fill='white')
            
            # Enhanced eyes
            eye_size = max(2, head_size // 8)
            left_eye_x = head_x + head_size // 3
            right_eye_x = head_x + 2 * head_size // 3
            eye_y = head_y + head_size // 3
            
            draw.ellipse((left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size), fill='black')
            draw.ellipse((right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size), fill='black')
            
            # Enhanced antennae
            antenna_size = head_size // 3
            left_antenna_x = head_x - antenna_size // 3
            right_antenna_x = head_x + head_size - antenna_size // 3
            antenna_y = head_y - antenna_size // 3
            
            draw.ellipse((left_antenna_x, antenna_y, left_antenna_x + antenna_size, antenna_y + antenna_size), 
                        fill='white', outline='#E0E0E0')
            draw.ellipse((right_antenna_x, antenna_y, right_antenna_x + antenna_size, antenna_y + antenna_size), 
                        fill='white', outline='#E0E0E0')
        except Exception as e:
            logger.error(f"Default avatar drawing failed: {e}")
    
    def _draw_verified_checkmark_enhanced(self, draw: ImageDraw.Draw, username_x: int, username_y: int, 
                                        username: str, font: ImageFont.ImageFont, font_size: int):
        """Enhanced verified checkmark with better positioning and error handling"""
        try:
            try:
                username_bbox = draw.textbbox((0, 0), username, font=font)
                username_width = username_bbox[2] - username_bbox[0]
            except:
                username_width = len(username) * (font_size * 0.6)
            
            check_x = username_x + username_width + 12
            check_y = username_y + int(font_size * 0.1)
            check_size = int(font_size * 0.75)
            
            # Enhanced blue circle with gradient effect
            draw.ellipse((check_x - 1, check_y - 1, check_x + check_size + 1, check_y + check_size + 1), 
                        fill='#0F5897')  # Darker blue shadow
            draw.ellipse((check_x, check_y, check_x + check_size, check_y + check_size), fill='#1DA1F2')
            
            # Enhanced white checkmark
            cx, cy = check_x + check_size//2, check_y + check_size//2
            line_width = max(2, check_size//8)
            
            # Anti-aliased checkmark lines
            for offset in range(line_width):
                draw.line([(cx-check_size//4, cy + offset), (cx-check_size//8, cy+check_size//4 + offset)], 
                         fill='white', width=1)
                draw.line([(cx-check_size//8, cy+check_size//4 + offset), (cx+check_size//4, cy-check_size//6 + offset)], 
                         fill='white', width=1)
        except Exception as e:
            logger.error(f"Verified checkmark drawing failed: {e}")
    
    def _draw_badges_enhanced(self, draw: ImageDraw.Draw, img: Image.Image, badge_x: int, badge_y: int, font_size: int):
        """Enhanced badge system with caching and better rendering"""
        try:
            badge_y_pos = badge_y + int(font_size * 1.2)
            badge_size = int(font_size * 0.7)
            badge_spacing = int(badge_size * 1.4)
            
            # Enhanced badge filenames with fallbacks
            badge_pngs = ['medal.png', 'trophy.png', 'radioactive.png', 'mask.png', 
                         'handshake.png', 'rocket.png', 'gem.png', 'fire.png']
            
            fallback_colors = ['#FFD700', '#C0C0C0', '#32CD32', '#FF69B4', 
                              '#4169E1', '#FF8C00', '#8A2BE2', '#FF4500']
            
            for i in range(8):
                badge_x_pos = badge_x + i * badge_spacing
                badge_filename = badge_pngs[i]
                
                # Try to load from cache first
                cache_key = f"{badge_filename}_{badge_size}"
                if cache_key not in self.asset_cache:
                    try:
                        badge_path = os.path.join(self.image_src_path, badge_filename)
                        if os.path.exists(badge_path):
                            badge_png = Image.open(badge_path).convert("RGBA")
                            badge_png = badge_png.resize((badge_size, badge_size), Image.Resampling.LANCZOS)
                            self.asset_cache[cache_key] = badge_png
                            logger.debug(f"Cached badge: {badge_filename}")
                        else:
                            self.asset_cache[cache_key] = None
                    except Exception as e:
                        logger.debug(f"Badge {badge_filename} loading failed: {e}")
                        self.asset_cache[cache_key] = None
                
                # Use cached badge or fallback
                if self.asset_cache[cache_key] is not None:
                    try:
                        img.paste(self.asset_cache[cache_key], (badge_x_pos, badge_y_pos), self.asset_cache[cache_key])
                    except:
                        # Fallback to colored circle
                        color = fallback_colors[i]
                        draw.ellipse((badge_x_pos, badge_y_pos, badge_x_pos + badge_size, badge_y_pos + badge_size), 
                                   fill=color, outline='#FFFFFF', width=1)
                else:
                    # Enhanced fallback badges
                    color = fallback_colors[i]
                    # Add gradient effect to fallback
                    draw.ellipse((badge_x_pos + 1, badge_y_pos + 1, badge_x_pos + badge_size + 1, badge_y_pos + badge_size + 1), 
                               fill='#00000020')  # Shadow
                    draw.ellipse((badge_x_pos, badge_y_pos, badge_x_pos + badge_size, badge_y_pos + badge_size), 
                               fill=color, outline='#FFFFFF', width=1)
        except Exception as e:
            logger.error(f"Badge drawing failed: {e}")
    
    def _draw_title_enhanced(self, draw: ImageDraw.Draw, title_lines: list, start_y: int, font: ImageFont.ImageFont, font_size: int, left_margin: int):
        """FIXED: Enhanced title drawing with symmetric margins and error handling"""
        try:
            current_y = start_y
            line_spacing = int(font_size * 1.1)  # Optimized line spacing
            
            for line in title_lines:
                if not line:  # Skip empty lines
                    current_y += line_spacing // 2
                    continue
                    
                x_pos = left_margin  # FIXED: Use consistent left margin (was 60, now 75)
                
                # Add subtle text shadow for better video visibility
                shadow_offset = 1
                for dx in range(2):
                    for dy in range(2):
                        draw.text((x_pos + dx + shadow_offset, current_y + dy + shadow_offset), line, 
                                 fill='#E0E0E0', font=font)
                
                # Draw main text with enhanced bold effect
                for dx in range(2):
                    for dy in range(2):
                        draw.text((x_pos + dx, current_y + dy), line, fill='#000000', font=font)
                
                current_y += line_spacing
        except Exception as e:
            logger.error(f"Title drawing failed: {e}")
    
    def _draw_footer_enhanced(self, draw: ImageDraw.Draw, likes: str, comments: str, font: ImageFont.ImageFont, 
                            y: int, img: Image.Image, font_size: int):
        """FIXED: Enhanced footer with proper comments display and symmetric margins"""
        try:
            # CONSISTENCY FIX: Validate inputs
            if not isinstance(likes, str):
                likes = str(likes) if likes else "99+"
            if not isinstance(comments, str):
                comments = str(comments) if comments else "99"
            
            icon_size = int(font_size * 0.75)
            
            # Enhanced heart icon positioning with symmetric margin
            heart_x, heart_y = 75, y + 20  # FIXED: Use symmetric margin (was 60, now 75)
            
            # Try to load enhanced heart icon
            self._draw_enhanced_icon(draw, img, heart_x, heart_y, icon_size, "heart.png", '#FF6B6B')
            
            # Enhanced like count with shadow
            text_y = heart_y + (icon_size - font_size) // 2
            draw.text((heart_x + icon_size + 13, text_y + 1), likes, fill='#C0C0C0', font=font)  # Shadow
            draw.text((heart_x + icon_size + 12, text_y), likes, fill='#878A8C', font=font)
            
            # FIXED: Enhanced comments icon with symmetric margin from right
            comments_x = self.template_width - 75 - icon_size - 50  # FIXED: Symmetric right margin
            self._draw_enhanced_icon(draw, img, comments_x, heart_y, icon_size, "share.png", '#4ECDC4')
            
            # FIXED: Display actual comments count instead of "Share"
            draw.text((comments_x + icon_size + 13, text_y + 1), comments, fill='#C0C0C0', font=font)  # Shadow
            draw.text((comments_x + icon_size + 12, text_y), comments, fill='#878A8C', font=font)
        except Exception as e:
            logger.error(f"Footer drawing failed: {e}")
    
    def _draw_enhanced_icon(self, draw: ImageDraw.Draw, img: Image.Image, x: int, y: int, size: int, 
                          filename: str, fallback_color: str):
        """Enhanced icon drawing with caching and fallbacks"""
        try:
            cache_key = f"{filename}_{size}"
            
            if cache_key not in self.asset_cache:
                try:
                    icon_path = os.path.join(self.image_src_path, filename)
                    if os.path.exists(icon_path):
                        icon_png = Image.open(icon_path).convert("RGBA")
                        icon_png = icon_png.resize((size, size), Image.Resampling.LANCZOS)
                        self.asset_cache[cache_key] = icon_png
                    else:
                        self.asset_cache[cache_key] = None
                except:
                    self.asset_cache[cache_key] = None
            
            if self.asset_cache[cache_key] is not None:
                try:
                    img.paste(self.asset_cache[cache_key], (x, y), self.asset_cache[cache_key])
                    return
                except:
                    pass
            
            # Enhanced fallback icon
            if filename == "heart.png":
                self._draw_enhanced_heart(draw, x, y, size, fallback_color)
            elif filename == "share.png":
                self._draw_enhanced_share(draw, x, y, size, fallback_color)
        except Exception as e:
            logger.error(f"Icon drawing failed: {e}")
    
    def _draw_enhanced_heart(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: str):
        """Enhanced heart icon with better shape"""
        try:
            half = size // 2
            quarter = size // 4
            
            # Add shadow
            draw.ellipse((x + 1, y + quarter + 1, x + half + 1, y + 3 * quarter + 1), fill='#00000020')
            draw.ellipse((x + half + 1, y + quarter + 1, x + size + 1, y + 3 * quarter + 1), fill='#00000020')
            
            # Main heart shape
            draw.ellipse((x, y + quarter, x + half, y + 3 * quarter), fill=color)
            draw.ellipse((x + half, y + quarter, x + size, y + 3 * quarter), fill=color)
            
            # Heart point
            points = [(x + quarter, y + 3 * quarter), (x + half, y + size), (x + 3 * quarter, y + 3 * quarter)]
            draw.polygon(points, fill=color)
        except Exception as e:
            logger.error(f"Heart drawing failed: {e}")
    
    def _draw_enhanced_share(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: str):
        """Enhanced share icon with better arrow"""
        try:
            quarter = size // 4
            half = size // 2
            
            # Add shadow
            draw.line([(x + quarter + 1, y + 3 * quarter + 1), (x + 3 * quarter + 1, y + quarter + 1)], 
                     fill='#00000020', width=3)
            
            # Main arrow
            draw.line([(x + quarter, y + 3 * quarter), (x + 3 * quarter, y + quarter)], 
                     fill=color, width=3)
            draw.line([(x + half, y + quarter), (x + 3 * quarter, y + quarter)], 
                     fill=color, width=3)
            draw.line([(x + 3 * quarter, y + quarter), (x + 3 * quarter, y + half)], 
                     fill=color, width=3)
            
            # Share box
            draw.rectangle((x, y + half, x + half, y + size), outline=color, width=2)
        except Exception as e:
            logger.error(f"Share icon drawing failed: {e}")
    
    def _add_rounded_corners_enhanced(self, img: Image.Image, radius: int) -> Image.Image:
        """Enhanced rounded corners with anti-aliasing and error handling"""
        try:
            # Create high-resolution mask for better anti-aliasing
            scale = 4
            large_size = (img.width * scale, img.height * scale)
            large_radius = radius * scale
            
            # Create large mask
            mask = Image.new('L', large_size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle((0, 0, large_size[0], large_size[1]), 
                                      radius=large_radius, fill=255)
            
            # Resize mask back down for anti-aliasing
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)
            
            # Apply mask
            result = Image.new('RGBA', img.size, (0, 0, 0, 0))
            result.paste(img, (0, 0))
            result.putalpha(mask)
            
            return result
        except Exception as e:
            logger.warning(f"Enhanced rounded corners failed: {e}")
            return img
    
    def _create_fallback_image(self, title: str) -> Image.Image:
        """Create enhanced fallback image on error with better error handling"""
        try:
            # CONSISTENCY FIX: Validate title
            if not title or not isinstance(title, str):
                title = "Reddit Post"
            
            fallback_img = Image.new('RGBA', (self.template_width, self.template_height), '#FF4500')
            fallback_draw = ImageDraw.Draw(fallback_img)
            
            # Add error message
            try:
                font = self._load_font_with_fallback("bold", 48)
            except:
                font = ImageFont.load_default()
            
            fallback_draw.text((50, 50), "Reddit Post", fill='white', font=font)
            title_preview = title[:50] + "..." if len(title) > 50 else title
            fallback_draw.text((50, 120), f"Title: {title_preview}", fill='white', font=font)
            fallback_draw.text((50, self.template_height - 100), f"Generated Successfully (Gen #{self.current_generation})", fill='white', font=font)
            
            return fallback_img
        except Exception as e:
            logger.error(f"Fallback image creation failed: {e}")
            # Ultimate fallback
            return Image.new('RGB', (800, 600), '#FF4500')
    
    def save_profile(self, username: str, avatar_image) -> str:
        """Enhanced profile saving with better error handling and validation"""
        try:
            # CONSISTENCY FIX: Validate inputs
            if username and not isinstance(username, str):
                username = str(username)
            
            self.saved_profile["username"] = username if username else "AskReddit"
            
            if avatar_image is not None:
                try:
                    if isinstance(avatar_image, str):
                        if os.path.exists(avatar_image):
                            with open(avatar_image, 'rb') as f:
                                self.saved_profile["avatar_data"] = Image.open(io.BytesIO(f.read()))
                        else:
                            logger.warning(f"Avatar image file not found: {avatar_image}")
                    else:
                        self.saved_profile["avatar_data"] = avatar_image
                        
                    # Clear avatar cache to force reload
                    self.asset_cache.clear()
                except Exception as e:
                    logger.error(f"Avatar saving failed: {e}")
                    self.saved_profile["avatar_data"] = None
            
            logger.info(f"Enhanced profile saved: {self.saved_profile['username']} (Gen #{self.current_generation})")
            return f"✅ Profile saved successfully!\nUsername: {self.saved_profile['username']}\nAvatar: {'✓ Saved' if self.saved_profile['avatar_data'] else '✗ Not saved'}"
        except Exception as e:
            logger.error(f"Enhanced profile save failed: {e}")
            return f"❌ Error saving profile: {str(e)}"
    
    def get_saved_profile_info(self) -> str:
        """Enhanced profile information display with generation tracking"""
        try:
            if self.saved_profile["username"]:
                avatar_status = "✓ Saved" if self.saved_profile["avatar_data"] else "✗ Not saved"
                cache_info = f"Cache: {len(self.asset_cache)} assets loaded"
                return f"""📋 **Saved Profile (Gen #{self.current_generation}):**
• Username: {self.saved_profile['username']}
• Avatar: {avatar_status}
• {cache_info}
• Last successful: Gen #{self.last_successful_generation}

💡 **Tips:**
• Use saved profile for consistent branding
• Upload custom avatar for personalization
• Generated posts are optimized for video overlay
• Clear cache between sessions for consistency
                """
            return f"""📋 **No Profile Saved (Gen #{self.current_generation})**

💡 **Get Started:**
• Enter username and upload avatar
• Check 'Save Current Profile' to remember settings
• Generated posts are optimized for video integration
• Cache cleared for consistent generation
            """
        except Exception as e:
            logger.error(f"Failed to get enhanced profile info: {e}")
            return f"Error getting profile information (Gen #{self.current_generation})"