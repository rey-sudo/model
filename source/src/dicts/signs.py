from pathlib import Path
from dicts.hashing import string_to_coords_3d
from dicts.codec import create_canvas_row
import re

class SignManager:
    def __init__(self):
        self.LSIGN = {}
        
    def get_coords_from_sign(self, sign: str, append=True) -> tuple[float, float, float]:
        """
        Returns deterministic 3D coordinates according to the linguistic sign and adds the sign to the idempotent dictionary
        """
        coords = string_to_coords_3d(sign)
        
        if append:
            self.LSIGN[coords] = sign
            
        return coords
              
    def get_sign_from_coords(self, coords: tuple[float, float, float]) -> str:
        """
        The linguistic sign returns from its deterministic coordinates.
        """
        sign = self.LSIGN.get(coords)
        return sign
    
    def clean_block(self, line: str):
        return re.findall(r'\b\d{4}\b|[a-zA-Z]{2,}', line)
    
    def apply_coords_to_block(self, array: list[str]) -> list[tuple[float, float, float]]:
        return [self.get_coords_from_sign(word.lower()) for word in array]
                
    def get_cascade_from_block(self, block: str):
        cleaned = self.clean_block(block)

        block = self.apply_coords_to_block(cleaned)
        return {i: block[:i+1] for i in range(len(block))}
    
    def decode_labels(self, label_str):
        # 1. Hacemos el split para obtener ['10', '11', '12']
        signs = label_str.split(',')
        
        # 2. Iteramos, convertimos a int y aplicamos get_sign_from_index
        # Asumiendo que get_sign_from_index recibe un entero
        resultado = [self.get_sign_from_index(int(idx)) for idx in signs]
        return " ".join(resultado)
    
    def load_block_file(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            return contenido
        except FileNotFoundError:
            return f"Error: El archivo en '{path}' no fue encontrado."
        except Exception as e:
            return f"Ocurrió un error inesperado: {e}"  
        
    def block_to_canvas(self, block: str, sign_size_px: int, total_signs: int):
        cleaned = self.clean_block(block)
        block = self.apply_coords_to_block(cleaned)
        
        canvas = create_canvas_row(value=block, sign_size_px=sign_size_px, total_signs=total_signs)
        return canvas
        
