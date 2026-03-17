from typing import Self
import src.other as c
from src.neuron.neuron import Neuron

neuron = Neuron()

class Like:
    def __init__(self, state: int):
        self.state = state
        self.x = state
        self.sign = "like"
        
    def like(self) -> Self:
        return self
            
    def coffe(self):
        return c.Coffe(1)
            
    def not_(self, a: int) -> int:
        return neuron.not_(a)   
    
    
    