from typing import Self
import src.other as c

class Fall:
    def __init__(self, state: int):
        self.state = state
        self.x = state
        
    def fall(self, a)-> Self:
        return self
    
    def down(self):
        return c.Down(1)
            
    def rain(self):
        return c.Rain(1)

    def up(self):
        return c.Up(0)   