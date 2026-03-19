from pathlib import Path
from typing import Any
from dicts.hashing import string_to_coords_3d
from dicts.codec import create_canvas_row
import re

class SignManager:
    def __init__(self):
        self.LSIGN = {}
        
    def _clean_block(self, line: str):
        return re.findall(r'\b\d{4}\b|[a-zA-Z]{2,}', line)
    
    def _map_coords(self, coord_list):
        return {i: tupla for i, tupla in enumerate(coord_list)}
    
    def _map_coords_reverse(self, coord_list):
        return {tupla: i for i, tupla in enumerate(coord_list)}
                
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
    
    def apply_coords_to_block(self, array: list[str]) -> list[tuple[float, float, float]]:
        return [self.get_coords_from_sign(word.lower()) for word in array]
                     
    def get_cascade_from_block(self, block: str):
        cleaned = self._clean_block(block)
        block_coords = self.apply_coords_to_block(cleaned)
        
        mapped_by_indices = self._map_coords(block_coords)
        mapped_by_coords = self._map_coords_reverse(block_coords)
        
        cascade = {i: list(range(i + 1)) for i in range(len(mapped_by_indices))}
    
        return mapped_by_coords, cascade
    
    def load_block_file(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            return contenido
        except FileNotFoundError:
            return f"Error: El archivo en '{path}' no fue encontrado."
        except Exception as e:
            return f"Ocurrió un error inesperado: {e}"  
        
    def _get_indices_from_smap(self, lista_tuplas, smap):
        return [smap.get(tupla) for tupla in lista_tuplas]
        
    def block_to_canvas(self, block: str, smap: dict[Any, int], sign_size_px: int, total_signs: int):
        cleaned = self._clean_block(block)
        block_coords = self.apply_coords_to_block(cleaned)
        
        values = self._get_indices_from_smap(block_coords, smap)
        
        #if none implement
        
        canvas = create_canvas_row(value=values, sign_size_px=sign_size_px, total_signs=total_signs)
        return canvas
        
    def decode_labels(self, ranking_label, smap):
        smap_keys = list(smap.keys())
        
        resultado = [self.LSIGN[smap_keys[int(n)]] for n in ranking_label.split(",")]
             
        return " ".join(resultado)
