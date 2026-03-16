from typing import Self


class Down:
    def __init__(self, state: int):
     self.state = state
     self.x = state
     
    def down(self) -> Self:
        return self     