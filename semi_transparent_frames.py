############################################# Semi-Transparent Frame Implementation ################################################

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import cv2

class SemiTransparentFrame:
    """
    A semi-transparent frame implementation using Canvas and PIL for RGBA support.
    Allows background images to show through with proper transparency effects.
    """
    
    def __init__(self, parent, x, y, width, height, alpha=0.7, color=(236, 240, 241), blur_radius=2):
        """
        Initialize semi-transparent frame
        
        Args:
            parent: Parent widget
            x, y: Position coordinates
            width, height: Dimensions
            alpha: Transparency (0.0 = fully transparent, 1.0 = fully opaque)
            color: RGB color tuple for the frame
            blur_radius: Blur effect radius for glass-like appearance
        """
        self.parent = parent
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alpha = alpha
        self.color = color
        self.blur_radius = blur_radius
        
        # Create the canvas that will hold our semi-transparent frame
        self.canvas = tk.Canvas(parent, highlightthickness=0, bd=0)
        self.canvas.place(x=x, y=y, width=width, height=height)
        
        # Create the semi-transparent image
        self.create_transparent_background()
        
        # Store child widgets for proper layering
        self.child_widgets = []
    
    def create_transparent_background(self):
        """Create a semi-transparent background with glass effect"""
        # Create base RGBA image
        rgba_image = Image.new('RGBA', (self.width, self.height), (*self.color, int(255 * self.alpha)))
        
        # Add subtle gradient effect for depth
        gradient = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(gradient)
        
        # Create vertical gradient
        for i in range(self.height):
            alpha_val = int(20 * (1 - i / self.height))  # Fade from top to bottom
            draw.rectangle([0, i, self.width, i+1], fill=(255, 255, 255, alpha_val))
        
        # Blend gradient with base
        rgba_image = Image.alpha_composite(rgba_image, gradient)
        
        # Add glass-like border effect
        border_overlay = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        border_draw = ImageDraw.Draw(border_overlay)
        
        # Top highlight
        border_draw.rectangle([0, 0, self.width, 2], fill=(255, 255, 255, 40))
        # Left highlight  
        border_draw.rectangle([0, 0, 2, self.height], fill=(255, 255, 255, 40))
        # Bottom shadow
        border_draw.rectangle([0, self.height-2, self.width, self.height], fill=(0, 0, 0, 30))
        # Right shadow
        border_draw.rectangle([self.width-2, 0, self.width, self.height], fill=(0, 0, 0, 30))
        
        # Apply border effects
        rgba_image = Image.alpha_composite(rgba_image, border_overlay)
        
        # Apply slight blur for glass effect
        if self.blur_radius > 0:
            rgba_image = rgba_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # Convert to PhotoImage for tkinter
        self.bg_image = ImageTk.PhotoImage(rgba_image)
        
        # Display on canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg_image)
        
        # Enable transparency for the canvas
        self.canvas.configure(bg='', highlightthickness=0)
    
    def add_widget(self, widget_class, **kwargs):
        """Add a widget to this semi-transparent frame"""
        # Create widget as child of the canvas
        widget = widget_class(self.canvas, **kwargs)
        self.child_widgets.append(widget)
        return widget
    
    def place_widget(self, widget, x, y, **kwargs):
        """Place a widget within this frame"""
        widget.place(x=x, y=y, **kwargs)
    
    def configure_transparency(self, alpha):
        """Update the transparency level"""
        self.alpha = alpha
        self.create_transparent_background()
    
    def update_size(self, width, height):
        """Update frame size"""
        self.width = width
        self.height = height
        self.canvas.place_configure(width=width, height=height)
        self.create_transparent_background()


class GlassFrame:
    """
    Enhanced glass-like frame with advanced visual effects
    """
    
    def __init__(self, parent, x, y, width, height, alpha=0.8, 
                 base_color=(240, 248, 255), accent_color=(100, 149, 237)):
        """
        Initialize glass frame with advanced effects
        
        Args:
            parent: Parent widget
            x, y: Position coordinates  
            width, height: Dimensions
            alpha: Base transparency
            base_color: Main frame color (RGB)
            accent_color: Accent color for highlights (RGB)
        """
        self.parent = parent
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alpha = alpha
        self.base_color = base_color
        self.accent_color = accent_color
        
        # Create canvas
        self.canvas = tk.Canvas(parent, highlightthickness=0, bd=0)
        self.canvas.place(x=x, y=y, width=width, height=height)
        
        self.create_glass_effect()
    
    def create_glass_effect(self):
        """Create sophisticated glass-like visual effect with enhanced pastel support"""
        # Create base layer with transparency
        base_layer = Image.new('RGBA', (self.width, self.height), 
                              (*self.base_color, int(255 * self.alpha)))
        
        # Create soft pastel gradient overlay
        gradient_layer = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        gradient_draw = ImageDraw.Draw(gradient_layer)
        
        # Add vertical gradient for depth
        for i in range(self.height):
            progress = i / self.height
            # Subtle gradient from light to slightly darker
            alpha_val = int(15 * (1 - progress * 0.5))
            gradient_draw.rectangle([0, i, self.width, i+1], 
                                  fill=(255, 255, 255, alpha_val))
        
        # Create reflection layer with softer effect
        reflection_layer = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        reflection_draw = ImageDraw.Draw(reflection_layer)
        
        # Add diagonal gradient reflection (softer for pastels)
        for i in range(min(self.width, self.height) // 4):
            alpha_val = int(20 * (1 - i / (min(self.width, self.height) // 4)))
            reflection_draw.polygon([
                (0, i), (i, 0), (i+2, 0), (0, i+2)
            ], fill=(255, 255, 255, alpha_val))
        
        # Create subtle texture (reduced for pastel theme)
        texture_layer = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        texture_draw = ImageDraw.Draw(texture_layer)
        
        # Add very subtle texture pattern
        import random
        for _ in range(self.width * self.height // 200):  # Reduced density
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            alpha_val = random.randint(2, 8)  # More subtle
            texture_draw.point((x, y), fill=(255, 255, 255, alpha_val))
        
        # Apply blur to texture for smooth appearance
        texture_layer = texture_layer.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Create soft border effect
        border_layer = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        border_draw = ImageDraw.Draw(border_layer)
        
        # Soft border with accent color
        for i in range(2):
            alpha_val = int(25 * (1 - i / 2))
            border_draw.rectangle([i, i, self.width-i-1, self.height-i-1], 
                                outline=(*self.accent_color, alpha_val), width=1)
        
        # Combine all layers
        final_image = base_layer
        final_image = Image.alpha_composite(final_image, gradient_layer)
        final_image = Image.alpha_composite(final_image, reflection_layer)
        final_image = Image.alpha_composite(final_image, texture_layer)
        final_image = Image.alpha_composite(final_image, border_layer)
        
        # Convert to PhotoImage
        self.bg_image = ImageTk.PhotoImage(final_image)
        
        # Display on canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg_image)
    
    def add_widget(self, widget_class, **kwargs):
        """Add widget to glass frame"""
        widget = widget_class(self.canvas, **kwargs)
        return widget


class AnimatedGlassFrame:
    """
    Glass frame with subtle animation effects
    """
    
    def __init__(self, parent, x, y, width, height, alpha=0.75):
        self.parent = parent
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alpha = alpha
        self.animation_offset = 0
        
        # Create canvas
        self.canvas = tk.Canvas(parent, highlightthickness=0, bd=0)
        self.canvas.place(x=x, y=y, width=width, height=height)
        
        self.create_animated_background()
        self.start_animation()
    
    def create_animated_background(self):
        """Create background with animation-ready elements"""
        # Base semi-transparent layer
        base_layer = Image.new('RGBA', (self.width, self.height), 
                              (220, 230, 240, int(255 * self.alpha)))
        
        # Animated shimmer effect
        shimmer_layer = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        shimmer_draw = ImageDraw.Draw(shimmer_layer)
        
        # Create moving highlight
        shimmer_x = int(self.animation_offset % (self.width + 100)) - 50
        if 0 <= shimmer_x <= self.width:
            for i in range(20):
                alpha_val = int(15 * (1 - abs(i - 10) / 10))
                x_pos = shimmer_x + i - 10
                if 0 <= x_pos < self.width:
                    shimmer_draw.line([(x_pos, 0), (x_pos, self.height)], 
                                    fill=(255, 255, 255, alpha_val), width=1)
        
        # Combine layers
        final_image = Image.alpha_composite(base_layer, shimmer_layer)
        
        # Convert and display
        self.bg_image = ImageTk.PhotoImage(final_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg_image)
    
    def start_animation(self):
        """Start the shimmer animation"""
        self.animation_offset += 2
        self.create_animated_background()
        self.parent.after(50, self.start_animation)
    
    def add_widget(self, widget_class, **kwargs):
        """Add widget to animated frame"""
        widget = widget_class(self.canvas, **kwargs)
        return widget


class FrameFactory:
    """
    Factory class to create different types of semi-transparent frames
    """
    
    @staticmethod
    def create_basic_frame(parent, x, y, width, height, alpha=0.7, color=(236, 240, 241)):
        """Create basic semi-transparent frame"""
        return SemiTransparentFrame(parent, x, y, width, height, alpha, color)
    
    @staticmethod
    def create_glass_frame(parent, x, y, width, height, alpha=0.8):
        """Create glass-effect frame"""
        return GlassFrame(parent, x, y, width, height, alpha)
    
    @staticmethod
    def create_animated_frame(parent, x, y, width, height, alpha=0.75):
        """Create animated glass frame"""
        return AnimatedGlassFrame(parent, x, y, width, height, alpha)
    
    @staticmethod
    def create_frosted_frame(parent, x, y, width, height, alpha=0.6, blur_radius=3):
        """Create frosted glass effect frame"""
        return SemiTransparentFrame(parent, x, y, width, height, alpha, 
                                  color=(240, 248, 255), blur_radius=blur_radius)


# Utility functions for frame integration
def replace_frame_with_glass(original_frame, parent, frame_type="glass"):
    """
    Replace an existing tkinter Frame with a semi-transparent glass frame
    
    Args:
        original_frame: The original tk.Frame to replace
        parent: Parent widget
        frame_type: Type of glass frame ("basic", "glass", "animated", "frosted")
    """
    # Get original frame properties
    info = original_frame.place_info()
    if not info:
        info = original_frame.pack_info()
    if not info:
        info = original_frame.grid_info()
    
    # Extract positioning
    x = int(info.get('x', 0))
    y = int(info.get('y', 0)) 
    width = int(info.get('width', 200))
    height = int(info.get('height', 100))
    
    # Destroy original frame
    original_frame.destroy()
    
    # Create new glass frame based on type
    factory = FrameFactory()
    if frame_type == "basic":
        return factory.create_basic_frame(parent, x, y, width, height)
    elif frame_type == "glass":
        return factory.create_glass_frame(parent, x, y, width, height)
    elif frame_type == "animated":
        return factory.create_animated_frame(parent, x, y, width, height)
    elif frame_type == "frosted":
        return factory.create_frosted_frame(parent, x, y, width, height)
    else:
        return factory.create_glass_frame(parent, x, y, width, height)


def create_layered_background(parent, background_image_path, overlay_alpha=0.3):
    """
    Create a layered background with semi-transparent overlay
    
    Args:
        parent: Parent widget
        background_image_path: Path to background image
        overlay_alpha: Transparency of the overlay
    """
    try:
        # Load and resize background image
        bg_image = Image.open(background_image_path)
        parent_width = parent.winfo_reqwidth() or 800
        parent_height = parent.winfo_reqheight() or 600
        
        bg_image = bg_image.resize((parent_width, parent_height), Image.Resampling.LANCZOS)
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', (parent_width, parent_height), 
                           (255, 255, 255, int(255 * overlay_alpha)))
        
        # Combine background with overlay
        if bg_image.mode != 'RGBA':
            bg_image = bg_image.convert('RGBA')
        
        final_bg = Image.alpha_composite(bg_image, overlay)
        
        # Convert to PhotoImage and apply
        bg_photo = ImageTk.PhotoImage(final_bg)
        
        # Create background label
        bg_label = tk.Label(parent, image=bg_photo)
        bg_label.image = bg_photo  # Keep reference
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        return bg_label
        
    except Exception as e:
        print(f"Could not create layered background: {e}")
        return None


# Example usage and demonstration
if __name__ == "__main__":
    # Demo application
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Semi-Transparent Frames Demo")
    root.configure(bg="#2c3e50")
    
    # Create different types of semi-transparent frames
    factory = FrameFactory()
    
    # Basic semi-transparent frame
    basic_frame = factory.create_basic_frame(root, 50, 50, 300, 200, alpha=0.7)
    label1 = basic_frame.add_widget(tk.Label, text="Basic Semi-Transparent Frame", 
                                   bg="white", fg="black", font=("Arial", 12, "bold"))
    basic_frame.place_widget(label1, 20, 20)
    
    # Glass effect frame  
    glass_frame = factory.create_glass_frame(root, 400, 50, 300, 200, alpha=0.8)
    label2 = glass_frame.add_widget(tk.Label, text="Glass Effect Frame", 
                                   bg="white", fg="navy", font=("Arial", 12, "bold"))
    glass_frame.canvas.create_window(150, 50, window=label2)
    
    # Animated frame
    animated_frame = factory.create_animated_frame(root, 50, 300, 300, 200, alpha=0.75)
    label3 = animated_frame.add_widget(tk.Label, text="Animated Glass Frame", 
                                      bg="white", fg="darkblue", font=("Arial", 12, "bold"))
    animated_frame.canvas.create_window(150, 50, window=label3)
    
    # Frosted frame
    frosted_frame = factory.create_frosted_frame(root, 400, 300, 300, 200, 
                                               alpha=0.6, blur_radius=4)
    label4 = frosted_frame.add_widget(tk.Label, text="Frosted Glass Frame", 
                                     bg="white", fg="darkgreen", font=("Arial", 12, "bold"))
    frosted_frame.place_widget(label4, 20, 20)
    
    root.mainloop()
