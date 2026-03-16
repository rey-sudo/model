from typing import Self
import src.other as c

class Do:
    def __init__(self, state: int):
     self.state = state    
     self.x = state
     self.sign = "do" 
     
    def do(self) -> Self:
        return self
         
    def you(self):
        return c.You(1)
    
