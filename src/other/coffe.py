from typing import Self


class Coffe:
    def __init__(self, state: int):
     self.state = state    
     self.x = state 
     self.sign = "coffe" 

    def coffe(self) -> Self:
        return self