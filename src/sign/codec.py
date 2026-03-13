import numpy as np
from PIL import Image

"""
     C0  C1  C2  C3  C4  C5  C6  C7  C8
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
F0  │ M │ M │ M │ M │ M │ M │ M │ M │ M │  <-- Frame (255)
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
F1  │ M │ . │ . │ . │ . │ . │ . │ . │ M │  <-- Quiet Zone (0)
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
F2  │ M │ . │ D │ D │ D │ D │ D │ . │ M │  ┐
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  │
F3  │ M │ . │ D │ D │ D │ D │ D │ . │ M │  │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  │   Data Matrix 
F4  │ M │ . │ D │ D │ D │ D │ D │ . │ M │  │      (5x5)
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  │   [2:7, 2:7]
F5  │ M │ . │ D │ D │ D │ D │ D │ . │ M │  │
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤  │
F6  │ M │ . │ D │ D │ D │ D │ D │ . │ M │  ┘
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
F7  │ M │ . │ . │ . │ . │ . │ . │ . │ M │  <-- Quiet Zone (0)
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
F8  │ M │ M │ M │ M │ M │ M │ M │ M │ M │  <-- Frame (255)
    └───┴───┴───┴───┴───┴───┴───┴───┴───┘
"""
def index_to_sign(n):
    """
    Encodes an integer into a 9x9 pixel 'sign' or micro-tag.
    The core data is a 5x5 matrix with bit-shuffling for high visual entropy,
    surrounded by a 1-pixel frame and a quiet zone for machine vision.
    """    
    # 1. 5x5 Data Core (Bit Dispersion)
    # Apply a fixed XOR mask (seed) to prevent solid blocks in low numbers
    seed = 0x1555555 
    # Ensure the number fits within 25 bits and apply the mask
    prepared_n = (n ^ seed) & 0x1FFFFFF 
    # Convert to a 25-bit binary string and reverse it to disperse bit changes
    binary_str = format(prepared_n, '025b')[::-1]
    # Map binary bits to pixel intensities: 1 -> White (255), 0 -> Black (0)
    data_points = [255 if b == '1' else 0 for b in binary_str]
    # Reshape the flat list into a 5x5 NumPy array
    data_matrix = np.array(data_points, dtype=np.uint8).reshape((5, 5))
    
    # 2. Create 9x9 Canvas
    # Initialize a black canvas (0 intensity) which acts as the 'Quiet Zone'
    canvas = np.zeros((9, 9), dtype=np.uint8)
    
    # 3. Draw Decorative/Alignment Frame
    # Create a 1-pixel white border on the outermost edges
    canvas[0, :] = 255  # Top
    canvas[8, :] = 255  # Bottom
    canvas[:, 0] = 255  # Left
    canvas[:, 8] = 255  # Right
    
    # 4. Center the Data Core
    # Place the 5x5 data matrix at the center, leaving a 1-pixel buffer 
    # (quiet zone) between the data and the frame for better CV detection.
    canvas[2:7, 2:7] = data_matrix
    
    # Return the result as a PIL Image object in 8-bit grayscale mode ('L')
    return Image.fromarray(canvas, mode='L')


