import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time

class TextCache:
    _cache = {}
    _font_cache = {}

    @classmethod
    def get_font(cls, size):
        font_key = f"arial_{size}"
        if font_key not in cls._font_cache:
            try:
                # Common Windows path - prioritized for Vietnamese support
                font_path = "C:/Windows/Fonts/arial.ttf" 
                if not os.path.exists(font_path):
                    font_path = "arial.ttf"
                cls._font_cache[font_key] = ImageFont.truetype(font_path, size)
            except:
                cls._font_cache[font_key] = ImageFont.load_default()
        return cls._font_cache[font_key]

    @classmethod
    def get_text_image(cls, text, size, color_bgr, outline_color=(0,0,0), outline_width=0):
        # Key includes all visual properties
        key = (text, size, color_bgr, outline_color, outline_width)
        
        if key in cls._cache:
            return cls._cache[key]
        
        # Render clean text using PIL
        font = cls.get_font(size)
        
        # Calculate size using getbox/getlength
        left, top, right, bottom = font.getbbox(text)
        text_w = right - left
        text_h = bottom - top
        
        # Add padding (more for outline)
        pad = outline_width * 2 + 10
        w = text_w + pad
        h = text_h + pad
        
        # Create transparent image
        img_pil = Image.new('RGBA', (w, h + 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img_pil)
        
        # Convert BGR to RGB
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        outline_rgb = (outline_color[2], outline_color[1], outline_color[0])
        
        # Draw text with outline
        # Position with padding
        x, y = pad // 2, 0
        if outline_width > 0:
            draw.text((x, y), text, font=font, fill=color_rgb + (255,), stroke_width=outline_width, stroke_fill=outline_rgb + (255,))
        else:
            draw.text((x, y), text, font=font, fill=color_rgb + (255,))
        
        # Convert to OpenCV (RGBA)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
        
        # Store in cache
        cls._cache[key] = img_cv
        return img_cv

class GameRenderer:
    def __init__(self):
        pass

    def draw_hand_cursor(self, frame, x, y, color=(0, 255, 255)):
        """
        Draws a large, glowing circle at the hand position.
        """
        # Outer glow - Reduced size significantly
        cv2.circle(frame, (x, y), 8, color, 2)
        cv2.circle(frame, (x, y), 5, color, 2)
        # Inner core
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        return frame

    def draw_text(self, frame, text, x, y, size, color=(255, 255, 255), outline_color=(0,0,0), outline_width=0):
        """
        Draws high-quality cached text onto the frame.
        """
        text_img = TextCache.get_text_image(text, size, color, outline_color, outline_width)
        h, w = text_img.shape[:2]
        
        # Overlay
        y1, y2 = y, y + h
        x1, x2 = x, x + w
        
        if y1 < 0: y1 = 0
        if x1 < 0: x1 = 0
        if y2 > frame.shape[0]: y2 = frame.shape[0]
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        
        # Calculate overlap dimensions
        overlay_h = y2 - y1
        overlay_w = x2 - x1
        
        if overlay_h <= 0 or overlay_w <= 0: return frame
        
        # Extract relevant part of the text image
        text_part = text_img[:overlay_h, :overlay_w]
        
        # Alpha blending
        alpha_s = text_part[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * text_part[:, :, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
        return frame

    def draw_shadow(self, frame, x, y, size=30):
        """
        Draws a semi-transparent shadow below the object.
        """
        # Create a shadow overlay
        shadow_h, shadow_w = 20, size * 2
        
        # Check bounds
        if y + 30 >= frame.shape[0] or x - shadow_w//2 < 0 or x + shadow_w//2 >= frame.shape[1]:
            return frame
            
        overlay = frame.copy()
        cv2.ellipse(overlay, (x, y + 40), (size, 10), 0, 0, 360, (0, 0, 0), -1)
        
        # Blend (Alpha 0.3)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame

    def overlay_image(self, background, overlay, x, y, angle=0):
        """
        Efficiently overlays a transparent image (optionally rotated).
        """
        h, w = overlay.shape[:2]
        
        if angle != 0:
            # Rotation
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - w // 2
            M[1, 2] += (nH / 2) - h // 2
            overlay = cv2.warpAffine(overlay, M, (nW, nH))
            h, w = overlay.shape[:2]

        if overlay.shape[2] < 4: return background # Must be BGRA
        
        # Centered coordinates
        x1, y1 = int(x - w // 2), int(y - h // 2)
        x2, y2 = x1 + w, y1 + h
        
        # Clipping
        bg_h, bg_w = background.shape[:2]
        
        # Calculate overlap
        ox1 = max(0, x1)
        oy1 = max(0, y1)
        ox2 = min(bg_w, x2)
        oy2 = min(bg_h, y2)
        
        if ox1 >= ox2 or oy1 >= oy2: return background
        
        # Offsets into overlay
        tx1 = ox1 - x1
        ty1 = oy1 - y1
        tx2 = tx1 + (ox2 - ox1)
        ty2 = ty1 + (oy2 - oy1)
        
        overlay_crop = overlay[ty1:ty2, tx1:tx2]
        bg_crop = background[oy1:oy2, ox1:ox2]
        
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        for c in range(3):
            bg_crop[:, :, c] = alpha * overlay_crop[:, :, c] + alpha_inv * bg_crop[:, :, c]
            
        background[oy1:oy2, ox1:ox2] = bg_crop
        return background

    def draw_half_fruit(self, background, img, x, y, angle, side='left'):
        """
        Draws half of the fruit image, rotated.
        side: 'left' or 'right'
        """
        if img is None: return background
        
        h, w = img.shape[:2]
        
        # Crop half
        mid_x = w // 2
        if side == 'left':
            half_img = img[:, :mid_x]
        else:
            half_img = img[:, mid_x:]
            
        # Check if empty
        if half_img.size == 0: return background
            
        # Overlay the half image
        # Center of rotation should still be somewhat central to the visual
        # But for simplicity, we treat the half image as a new sprite
        return self.overlay_image(background, half_img, x, y, angle)

    def draw_particles(self, frame, particles):
        for p in particles:
            if p.life <= 0: continue
            
            # Draw splash-like particle
            # Use ellipse for directional splash if we had velocity, but circle is fine
            # Maybe vary size slightly by life?
            size = p.size
            if p.is_splash:
                # Shrink correctly over life
                pass
            
            cv2.circle(frame, (int(p.x), int(p.y)), size, p.color, -1)
        return frame

    def draw_trail(self, frame, points, color=(0, 255, 255)):
        """
        Draws a glowing trail.
        """
        if len(points) < 2: return frame
        
        # Draw Glow (Thick, colored)
        # We can simulate glow with multiple lines or just one thick line
        # Simpler is better for performance
        
        # 1. Outer Glow
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            progress = i / (len(points) - 1)
            thickness = int(4 + 12 * progress) # Thicker
            
            # Fade alpha? CV2 line doesn't support alpha directly without overly complex blending
            # Just draw solid color for now, maybe BGR
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
            
        # 2. Inner Core (White/Bright)
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            progress = i / (len(points) - 1)
            thickness = int(1 + 4 * progress) # Thinner
            
            cv2.line(frame, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)
            
        return frame

    def draw_button(self, frame, text, x, y, w, h, hover=False):
        """
        Draws a button with text.
        """
        color = (0, 255, 0) if hover else (0, 200, 200) # Green if active, else Cyan
        # Draw Rect
        cv2.rectangle(frame, (int(x - w//2), int(y - h//2)), (int(x + w//2), int(y + h//2)), color, -1)
        cv2.rectangle(frame, (int(x - w//2), int(y - h//2)), (int(x + w//2), int(y + h//2)), (255, 255, 255), 2)
        
        # Draw Text using TextCache for consistent font support
        text_img = TextCache.get_text_image(text, 30, (0, 0, 0)) # Black text
        img_h, img_w = text_img.shape[:2]
        
        # Center text image on button
        bx = int(x - img_w//2)
        by = int(y - img_h//2)
        
        # Overlay using existing helper
        # We need to ensure alpha blending works. TextCache returns BGRA.
        # Button background is already drawn on 'frame'.
        
        # Manual overlay for speed/simplicity since it's just a copy if no rotation
        if img_h > 0 and img_w > 0:
            y1, y2 = by, by + img_h
            x1, x2 = bx, bx + img_w
            
            # Clip
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            if y2 > frame.shape[0]: y2 = frame.shape[0]
            if x2 > frame.shape[1]: x2 = frame.shape[1]
            
            # Recalculate dimensions after clip
            h_clip = y2 - y1
            w_clip = x2 - x1
            
            if h_clip > 0 and w_clip > 0:
                # Text sprite crop
                ts_y1 = 0 + (y1 - by)
                ts_x1 = 0 + (x1 - bx)
                text_part = text_img[ts_y1:ts_y1+h_clip, ts_x1:ts_x1+w_clip]
                
                # Frame crop
                frame_part = frame[y1:y2, x1:x2]
                
                # Alpha blend
                alpha_s = text_part[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                
                for c in range(3):
                    frame_part[:, :, c] = (alpha_s * text_part[:, :, c] +
                                          alpha_l * frame_part[:, :, c])
                                          
                frame[y1:y2, x1:x2] = frame_part

        return frame
