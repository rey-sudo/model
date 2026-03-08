from typing import Self


class Rain:
    def __init__(self, state: int):
        self.state = state
        self.x = state

    def rain(self, a) -> Self:
        return self
            
    def fall(self):
        return Fall(1)
    
    def down(self):
        return Down(1)
              

class Fall:
    def __init__(self, state: int):
        self.state = state
        self.x = state
        
    def fall(self, a)-> Self:
        return self
    
    def down(self):
        return Down(1)
            
    def rain(self):
        return Rain(1)

    def up(self):
        return Up(0)        
        
class Down:
    def __init__(self, state: int):
     self.state = state
     self.x = state
     
class Up:
    def __init__(self, state: int):
     self.state = state    
     self.x = state   
        
rain = Rain(1)
fall = Rain(1)
down = Rain(1)


result = [rain.fall().x, rain.fall().up().x]

print(result)

#rain -> fall -> down

