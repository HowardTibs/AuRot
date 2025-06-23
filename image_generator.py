import gradio as gr
import PIL
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import os
from typing import Optional, Tuple, Dict

class RedditTemplateGenerator:
    def __init__(self):
        self.saved_profile = {
            "username": "AskReddit",
            "avatar": None,
            "avatar_data": None
        }
        self.template_width = 1800  # Keep original width
        self.template_height = 900  # Base height, will expand as needed
        
    def create_reddit_template(self, title: str, username: str = None, avatar_image = None, 
                             likes: str = "99+", comments: str = "99+", use_saved_profile: bool = False) -> PIL.Image.Image:
        """Create a Reddit post template with custom title, username, and avatar"""
        
        # Use saved profile if requested
        if use_saved_profile and username is None:
            username = self.saved_profile["username"]
        if use_saved_profile and avatar_image is None and self.saved_profile["avatar_data"]:
            avatar_image = self.saved_profile["avatar_data"]
            
        # Set defaults
        if username is None:
            username = "AskReddit"
        
        # Dynamic font sizing based on title length
        title_length = len(title)
        if title_length < 30:  # Short titles
            base_font_size = 94  # Slightly larger
        elif title_length < 60:  # Medium titles (current baseline)
            base_font_size = 84  # Keep current size
        elif title_length < 100:  # Long titles
            base_font_size = 74  # Smaller
        else:  # Very long titles
            base_font_size = 66  # Much smaller but still readable
            
        username_font_size = int(base_font_size * 0.8125)  # 68px baseline
        stats_font_size = int(base_font_size * 0.6875)    # 58px baseline
        
        # Calculate required height based on title length
        temp_img = Image.new('RGB', (self.template_width, 100), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        
        try:
            # Load fonts with proper sizing schematic
            title_font = ImageFont.truetype("arialbd.ttf", base_font_size)
            username_font = ImageFont.truetype("arialbd.ttf", username_font_size)
            stats_font = ImageFont.truetype("arial.ttf", stats_font_size)
        except:
            try:
                title_font = ImageFont.truetype("arial.ttf", base_font_size)
                username_font = ImageFont.truetype("arial.ttf", username_font_size)
                stats_font = ImageFont.truetype("arial.ttf", stats_font_size)
            except:
                try:
                    title_font = ImageFont.truetype("calibrib.ttf", base_font_size)
                    username_font = ImageFont.truetype("calibrib.ttf", username_font_size)
                    stats_font = ImageFont.truetype("calibri.ttf", stats_font_size)
                except:
                    # Fallback to default font
                    title_font = ImageFont.load_default()
                    username_font = ImageFont.load_default()
                    stats_font = ImageFont.load_default()
        
        # Calculate title dimensions with word wrapping
        title_area_width = self.template_width - 175  # Adjusted for reduced left margin
        title_lines = self._wrap_text(title, title_font, title_area_width, temp_draw)
        title_height = len(title_lines) * int(base_font_size * 1.2)  # Dynamic line height based on font size
        
        # Calculate total height needed
        header_height = 200
        title_margin = 60
        footer_height = 120
        total_height = header_height + title_margin + title_height + title_margin + footer_height
        
        # Create the main image
        img = Image.new('RGB', (self.template_width, total_height), '#FFFFFF')
        draw = ImageDraw.Draw(img)
        
        # Draw the header section (username area)
        self._draw_header(draw, username, avatar_image, username_font, img, username_font_size)
        
        # Draw the title section
        title_y = header_height + title_margin
        self._draw_title(draw, title_lines, title_y, title_font, base_font_size)
        
        # Draw the footer (stats)
        footer_y = total_height - footer_height
        self._draw_footer(draw, likes, comments, stats_font, footer_y, img, stats_font_size)
        
        # Add rounded corners to match template
        img = self._add_rounded_corners(img, 30)  # 30px corner radius
        
        return img
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.Draw) -> list:
        """Wrap text to fit within the specified width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, force it on its own line
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_header(self, draw: ImageDraw.Draw, username: str, avatar_image, font: ImageFont.ImageFont, img: Image.Image, username_font_size: int):
        """Draw the header section with avatar and username"""
        # Avatar size large enough to span both username AND badge rows (like template)
        avatar_size = 160  # Much larger to span both rows
        avatar_x, avatar_y = 75, 20  # Reduced left margin by about half
        
        # Handle avatar image
        avatar_loaded = False
        if avatar_image is not None:
            try:
                # Load and process the avatar image
                avatar = Image.open(avatar_image).convert('RGB')
                
                # Resize to fit perfectly in circle
                avatar = avatar.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)
                
                # Apply circular clipping by drawing pixel by pixel
                for y in range(avatar_size):
                    for x in range(avatar_size):
                        # Check if pixel is inside circle
                        center = avatar_size // 2
                        if (x - center) ** 2 + (y - center) ** 2 <= center ** 2:
                            pixel = avatar.getpixel((x, y))
                            # Draw this pixel
                            draw.rectangle((avatar_x + x, avatar_y + y, avatar_x + x + 1, avatar_y + y + 1), 
                                         fill=pixel)
                
                avatar_loaded = True
                print("✅ Avatar loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading avatar: {e}")
                avatar_loaded = False
        
        # Only draw default avatar if no image was provided OR loading failed
        if not avatar_loaded:
            self._draw_default_avatar(draw, avatar_x, avatar_y, avatar_size)
        
        # USERNAME and verified checkmark (positioned higher to align with larger avatar)
        username_x = avatar_x + avatar_size + 40
        username_y = avatar_y + 20  # Higher positioning for larger avatar
        draw.text((username_x, username_y), username, fill='#000000', font=font)
        
        # Draw verified checkmark next to username (scaled to username size)
        # Get the actual width of the username text
        username_bbox = draw.textbbox((0, 0), username, font=font)
        username_actual_width = username_bbox[2] - username_bbox[0]
        
        check_x = username_x + username_actual_width + 15  # 25% more spacing (12 + 3 = 15px)
        check_y = username_y + int(username_font_size * 0.1)  # Better alignment with text baseline
        check_size = int(username_font_size * 0.8)  # Scale with username
        
        # Draw blue circle background
        draw.ellipse((check_x, check_y, check_x + check_size, check_y + check_size), fill='#1DA1F2')
        
        # Draw white checkmark manually (scaled)
        cx, cy = check_x + check_size//2, check_y + check_size//2
        line_width = max(2, check_size//10)
        draw.line([(cx-check_size//4, cy), (cx-check_size//8, cy+check_size//4)], fill='white', width=line_width)
        draw.line([(cx-check_size//8, cy+check_size//4), (cx+check_size//4, cy-check_size//6)], fill='white', width=line_width)
        
        # PNG BADGES from image_src folder (exactly 8 badges)
        badge_x = username_x
        badge_y = username_y + int(username_font_size * 1.3)  # Better spacing below username
        
        # PNG badge filenames (8 badges using your actual filenames)
        badge_pngs = ['medal.png', 'trophy.png', 'radioactive.png', 'mask.png', 
                     'handshake.png', 'rocket.png', 'gem.png', 'fire.png']
        
        # Badge size to match template proportions
        badge_size = int(username_font_size * 0.8)  # Same size as before
        badge_spacing = int(badge_size * 1.5)  # Good spacing between badges
        
        # Draw PNG badges
        for i in range(8):  # Exactly 8 badges now
            badge_x_pos = badge_x + i * badge_spacing
            badge_filename = badge_pngs[i]
            
            try:
                # Load PNG badge from image_src folder
                badge_png = Image.open(f"image_src/{badge_filename}").convert("RGBA")
                badge_png = badge_png.resize((badge_size, badge_size), Image.Resampling.LANCZOS)
                
                # Paste the PNG badge onto the main image
                img.paste(badge_png, (badge_x_pos, badge_y), badge_png)
                print(f"✅ Badge {i+1} ({badge_filename}) loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading badge {badge_filename}: {e}")
                # Fallback: draw colored circle if PNG fails
                fallback_colors = ['#32CD32', '#FF69B4', '#FFD700', '#4169E1', 
                                 '#8A2BE2', '#FF8C00', '#DAA520', '#00CED1']
                color = fallback_colors[i % len(fallback_colors)]
                draw.ellipse((badge_x_pos, badge_y, badge_x_pos + badge_size, badge_y + badge_size), fill=color)
        
    def _draw_default_avatar(self, draw: ImageDraw.Draw, avatar_x: int, avatar_y: int, avatar_size: int):
        """Draw the default Reddit alien avatar"""
        # Orange circle background
        draw.ellipse((avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size), 
                    fill='#FF4500')
        
        # Reddit alien head (white circle)
        head_size = int(avatar_size * 0.4)
        head_x = avatar_x + (avatar_size - head_size) // 2
        head_y = avatar_y + int(avatar_size * 0.15)
        draw.ellipse((head_x, head_y, head_x + head_size, head_y + head_size), fill='white')
        
        # Alien eyes (black dots)
        eye_size = max(3, head_size // 10)
        left_eye_x = head_x + head_size // 4
        right_eye_x = head_x + 3 * head_size // 4 - eye_size
        eye_y = head_y + head_size // 3
        draw.ellipse((left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size), fill='black')
        draw.ellipse((right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size), fill='black')
        
        # Alien antennae (white circles)
        antenna_size = head_size // 4
        left_antenna_x = head_x - antenna_size // 2
        right_antenna_x = head_x + head_size - antenna_size // 2
        antenna_y = head_y - antenna_size // 2
        draw.ellipse((left_antenna_x, antenna_y, left_antenna_x + antenna_size, antenna_y + antenna_size), fill='white')
        draw.ellipse((right_antenna_x, antenna_y, right_antenna_x + antenna_size, antenna_y + antenna_size), fill='white')
        
        # Draw smile
        mouth_y = head_y + 2 * head_size // 3
        mouth_width = head_size // 3
        mouth_x_start = head_x + (head_size - mouth_width) // 2
        draw.line([(mouth_x_start, mouth_y), (mouth_x_start + mouth_width, mouth_y)], fill='black', width=2)
    
    def _draw_title(self, draw: ImageDraw.Draw, title_lines: list, start_y: int, font: ImageFont.ImageFont, base_font_size: int):
        """Draw the main title text in BOLD"""
        current_y = start_y
        line_spacing = int(base_font_size * 1.2)  # Dynamic line spacing
        for line in title_lines:
            # Create bold effect by drawing text multiple times with slight offsets
            x_pos = 75  # Reduced left margin to match avatar
            # Draw the text with multiple offset layers to create bold effect
            for dx in range(3):
                for dy in range(3):
                    draw.text((x_pos + dx, current_y + dy), line, fill='#000000', font=font)
            current_y += line_spacing
    
    def _draw_footer(self, draw: ImageDraw.Draw, likes: str, comments: str, font: ImageFont.ImageFont, y: int, img: Image.Image, stats_font_size: int):
        """Draw the footer with likes and share using PNG icons"""
        # Icon size based on stats font size (68.75% of main text)
        icon_size = int(stats_font_size * 0.8)  # Slightly smaller than text
        
        # Draw like button (heart icon) - LEFT SIDE
        heart_x, heart_y = 75, y + 25  # Reduced left margin to match other elements
        
        # Try to load heart PNG from image_src folder
        try:
            heart_icon = Image.open("image_src/heart.png").convert("RGBA")
            heart_icon = heart_icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            img.paste(heart_icon, (heart_x, heart_y), heart_icon)
            print("✅ Heart icon loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading heart icon: {e}")
            # Fallback: draw simple heart
            self._draw_simple_heart(draw, heart_x, heart_y, icon_size, '#878A8C')
        
        # Draw like count
        draw.text((heart_x + icon_size + 15, heart_y + (icon_size - stats_font_size) // 2), 
                 likes, fill='#878A8C', font=font)
        
        # Draw share button (arrow) - RIGHT SIDE
        share_x = self.template_width - 250  # Better positioning for 1800px width
        
        # Try to load share PNG from image_src folder
        try:
            share_icon = Image.open("image_src/share.png").convert("RGBA")
            share_icon = share_icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            img.paste(share_icon, (share_x, heart_y), share_icon)
            print("✅ Share icon loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading share icon: {e}")
            # Fallback: draw simple share arrow
            self._draw_simple_share(draw, share_x, heart_y, icon_size, '#878A8C')
        
        # Draw share count
        draw.text((share_x + icon_size + 15, heart_y + (icon_size - stats_font_size) // 2), 
                 "99+", fill='#878A8C', font=font)
    
    def _add_rounded_corners(self, img: Image.Image, radius: int) -> Image.Image:
        """Add rounded corners to the image"""
        # Create a mask with rounded corners
        mask = Image.new('L', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw a rounded rectangle mask
        mask_draw.rounded_rectangle(
            (0, 0, img.width, img.height),
            radius=radius,
            fill=255
        )
        
        # Create a new image with transparency
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(img, (0, 0))
        result.putalpha(mask)
        
        return result
    
    def _draw_simple_heart(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: str):
        """Fallback simple heart drawing"""
        # Simple heart outline
        half_size = size // 2
        quarter_size = size // 4
        
        # Left curve
        draw.ellipse((x, y + quarter_size, x + half_size, y + 3 * quarter_size), outline=color, width=2)
        # Right curve  
        draw.ellipse((x + half_size, y + quarter_size, x + size, y + 3 * quarter_size), outline=color, width=2)
        # Bottom point
        draw.line([(x + quarter_size, y + 3 * quarter_size), (x + half_size, y + size)], fill=color, width=2)
        draw.line([(x + half_size, y + size), (x + 3 * quarter_size, y + 3 * quarter_size)], fill=color, width=2)
    
    def _draw_simple_share(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: str):
        """Fallback simple share arrow drawing"""
        # Simple share arrow
        quarter_size = size // 4
        half_size = size // 2
        
        # Arrow shaft
        draw.line([(x + quarter_size, y + 3 * quarter_size), (x + 3 * quarter_size, y + quarter_size)], fill=color, width=3)
        # Arrow head
        draw.line([(x + half_size, y + quarter_size), (x + 3 * quarter_size, y + quarter_size)], fill=color, width=3)
        draw.line([(x + 3 * quarter_size, y + quarter_size), (x + 3 * quarter_size, y + half_size)], fill=color, width=3)
        # Box at bottom
        draw.rectangle((x, y + half_size, x + half_size, y + size), outline=color, width=2)
    
    def save_profile(self, username: str, avatar_image) -> str:
        """Save the current profile for future use"""
        self.saved_profile["username"] = username if username else "AskReddit"
        
        if avatar_image is not None:
            # Save avatar data
            if isinstance(avatar_image, str):
                with open(avatar_image, 'rb') as f:
                    self.saved_profile["avatar_data"] = Image.open(io.BytesIO(f.read()))
            else:
                self.saved_profile["avatar_data"] = avatar_image
        
        return f"Profile saved! Username: {self.saved_profile['username']}"
    
    def get_saved_profile_info(self) -> str:
        """Get information about the currently saved profile"""
        if self.saved_profile["username"]:
            avatar_status = "✓ Saved" if self.saved_profile["avatar_data"] else "✗ Not saved"
            return f"Saved Profile:\nUsername: {self.saved_profile['username']}\nAvatar: {avatar_status}"
        return "No profile saved"

# Initialize the generator
generator = RedditTemplateGenerator()

def generate_reddit_post(title, username, avatar, likes, comments, use_saved_profile, save_current_profile):
    """Generate Reddit post template"""
    if not title.strip():
        return None, "Please enter a title for the post"
    
    # Generate the image
    img = generator.create_reddit_template(
        title=title,
        username=username if username.strip() else None,
        avatar_image=avatar,
        likes=likes if likes.strip() else "99+",
        comments=comments if comments.strip() else "99+",
        use_saved_profile=use_saved_profile
    )
    
    # Save profile if requested
    status_message = ""
    if save_current_profile:
        save_msg = generator.save_profile(username, avatar)
        status_message = save_msg
    
    # Get profile info
    profile_info = generator.get_saved_profile_info()
    
    return img, f"{status_message}\n\n{profile_info}"

def clear_inputs():
    """Clear all input fields"""
    return "", "", None, "", "", False, False

# Create Gradio interface
with gr.Blocks(title="Reddit Post Template Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔥 Reddit Post Template Generator")
    gr.Markdown("Create authentic-looking Reddit posts with custom titles, usernames, and avatars!")
    gr.Markdown("**Note:** Place heart.png, share.png, and badge1.png through badge8.png in the 'image_src' folder for custom icons.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✏️ Post Content")
            title_input = gr.Textbox(
                label="Post Title",
                placeholder="Enter your Reddit post title here...",
                lines=3,
                max_lines=5
            )
            
            gr.Markdown("### 👤 User Profile")
            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter username (leave empty to use saved profile)",
                value=""
            )
            
            avatar_input = gr.File(
                label="Avatar Image",
                file_types=["image"],
                type="filepath"
            )
            
            gr.Markdown("### 📊 Post Stats")
            with gr.Row():
                likes_input = gr.Textbox(
                    label="Likes",
                    placeholder="99+",
                    value="99+",
                    scale=1
                )
                comments_input = gr.Textbox(
                    label="Comments", 
                    placeholder="99+",
                    value="99+",
                    scale=1
                )
            
            gr.Markdown("### ⚙️ Profile Settings")
            use_saved_profile = gr.Checkbox(
                label="Use Saved Profile",
                value=False,
                info="Use previously saved username and avatar"
            )
            
            save_current_profile = gr.Checkbox(
                label="Save Current Profile",
                value=False,
                info="Save current username and avatar for future use"
            )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Generate Reddit Post", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Post")
            output_image = gr.Image(
                label="Reddit Post",
                type="pil",
                interactive=False
            )
            
            status_output = gr.Textbox(
                label="Status & Profile Info",
                interactive=False,
                lines=4
            )
            
            gr.Markdown("### 💾 Download")
            gr.Markdown("Right-click on the generated image above to save it!")
    
    # Event handlers
    generate_btn.click(
        fn=generate_reddit_post,
        inputs=[title_input, username_input, avatar_input, likes_input, comments_input, use_saved_profile, save_current_profile],
        outputs=[output_image, status_output]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        outputs=[title_input, username_input, avatar_input, likes_input, comments_input, use_saved_profile, save_current_profile]
    )
    
    # Load profile info on startup
    app.load(
        fn=lambda: generator.get_saved_profile_info(),
        outputs=status_output
    )

if __name__ == "__main__":
    app.launch(debug=True)