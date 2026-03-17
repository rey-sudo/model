class Neuron:
    def __init__(self, state: int = 0):
        self.state = state
    
    def _validar(self, *args):
        if not all(v in (0, 1) for v in args):
            raise ValueError(f"Invalid Value: {args}")

    def nand(self, a, b):
        self._validar(a, b)
        return 1 - (a & b)

    def not_(self, a):
        self._validar(a)
        return self.nand(a, a)

    def and_(self, a, b):
        self._validar(a, b)
        return self.not_(self.nand(a, b))

    def or_(self, a, b):
        self._validar(a, b)
        return self.nand(self.not_(a), self.not_(b))

    def nor(self, a, b):
        self._validar(a, b)
        return self.not_(self.or_(a, b))

    def xor(self, a, b):
        self._validar(a, b)
        t1 = self.nand(a, b)
        t2 = self.nand(a, t1)
        t3 = self.nand(b, t1)
        return self.nand(t2, t3)

    def xnor(self, a, b):
        self._validar(a, b)
        return self.not_(self.xor(a, b))