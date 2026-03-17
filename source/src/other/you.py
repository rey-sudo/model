from typing import Self
import src.other as c

class You:
    def __init__(self, state: int):
     self.state = state    
     self.x = state
     self.sign = "you" 
     
    def you(self) -> Self:
        return self
         
    def like(self):
        return c.Like(1)
    
    def coffe(self):
        return c.Coffe(1)