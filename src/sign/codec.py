import numpy as np
from PIL import Image

"""
5x5 = 2^25 = 33,554,432

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

def sign_to_index(img):
    """
    Decodes a 9x9 pixel Image (PIL) back into the original integer index.
    Reverses the spatial centering, bit reversal, and XOR masking.
    """
    # 1. Convert PIL Image back to a NumPy array
    canvas = np.array(img, dtype=np.uint8)
    
    # 2. Extract the 5x5 Data Core
    # We slice the same coordinates [2:7, 2:7] used during encoding
    data_matrix = canvas[2:7, 2:7]
    
    # 3. Flatten the matrix and convert pixel intensities to bits
    # 255 (White) becomes '1', everything else (Black) becomes '0'
    flat_data = data_matrix.flatten()
    binary_list = ['1' if p == 255 else '0' for p in flat_data]
    
    # 4. Reverse the Bit Reversal
    # Join the list into a string and flip it back to its original order
    binary_str = "".join(binary_list)[::-1]
    
    # 5. Convert Binary String to Integer
    n_preparado = int(binary_str, 2)
    
    # 6. Reverse the XOR Mask
    # Applying XOR with the same seed restores the original value
    seed = 0x1555555
    original_index = n_preparado ^ seed
    
    return original_index

def block_to_canvas(acc, sign_size:int=0, block_length:int = 0):
    """
    Creates a (n x n) canvas and arranges signs (n x n) in a 
    triangular pattern from index 0 to n_max.
    """
    
    canva_size = sign_size * block_length
    # 1. Create a blank grayscale canvas
    atlas = Image.new('L', (canva_size, canva_size), 0)
    
    # 2. Iterate through rows con enumerate para tener la posición Y
    for y_idx, row in enumerate(acc):

        for col_val in row:
            # col_val es el ID o índice (0, 1, 2...)
            sign_img = index_to_sign(col_val)
            
            # La X depende del valor del elemento (col_val)
            # La Y depende de en qué nivel de la cascada estamos (y_idx)
            x_offset = col_val * sign_size
            y_offset = y_idx * sign_size

            atlas.paste(sign_img, (x_offset, y_offset))

    return atlas

