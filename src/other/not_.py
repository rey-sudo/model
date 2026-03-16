import src.other as c

class Not:
    def __init__(self, state: int):
     self.state = state    
     self.x = state 
     self.sign = "not"
     
    def fall(self):
        return c.Fall(1)     