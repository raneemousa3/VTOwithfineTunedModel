#!/usr/bin/env python3
"""
Create a test image for pose detection
"""

import cv2
import numpy as np
from pathlib import Path
import os

def create_test_image():
    """Create a simple test image with a person-like shape"""
    print("\n🎨 Creating test image...")
    
    # Create a blank image
    height, width = 600, 400
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a simple person shape
    # Head
    cv2.circle(image, (200, 100), 30, (100, 100, 100), -1)
    
    # Body
    cv2.rectangle(image, (170, 130), (230, 300), (150, 150, 150), -1)
    
    # Arms
    cv2.rectangle(image, (130, 150), (170, 250), (150, 150, 150), -1)
    cv2.rectangle(image, (230, 150), (270, 250), (150, 150, 150), -1)
    
    # Legs
    cv2.rectangle(image, (170, 300), (195, 450), (150, 150, 150), -1)
    cv2.rectangle(image, (205, 300), (230, 450), (150, 150, 150), -1)
    
    # Save the image
    os.makedirs("test_images/input", exist_ok=True)
    cv2.imwrite("test_images/input/test_person.jpg", image)
    print("✅ Created test image at test_images/input/test_person.jpg")
    return image

if __name__ == "__main__":
    create_test_image() 