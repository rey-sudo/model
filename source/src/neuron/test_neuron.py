import unittest
from src.neuron.neuron import Neuron

BITS = [(0, 0), (0, 1), (1, 0), (1, 1)]


class TestNand(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.nand(0, 0), 1)
    def test_01(self): self.assertEqual(self.p.nand(0, 1), 1)
    def test_10(self): self.assertEqual(self.p.nand(1, 0), 1)
    def test_11(self): self.assertEqual(self.p.nand(1, 1), 0)


class TestNot(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_0(self): self.assertEqual(self.p.not_(0), 1)
    def test_1(self): self.assertEqual(self.p.not_(1), 0)


class TestAnd(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.and_(0, 0), 0)
    def test_01(self): self.assertEqual(self.p.and_(0, 1), 0)
    def test_10(self): self.assertEqual(self.p.and_(1, 0), 0)
    def test_11(self): self.assertEqual(self.p.and_(1, 1), 1)


class TestOr(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.or_(0, 0), 0)
    def test_01(self): self.assertEqual(self.p.or_(0, 1), 1)
    def test_10(self): self.assertEqual(self.p.or_(1, 0), 1)
    def test_11(self): self.assertEqual(self.p.or_(1, 1), 1)


class TestNor(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.nor(0, 0), 1)
    def test_01(self): self.assertEqual(self.p.nor(0, 1), 0)
    def test_10(self): self.assertEqual(self.p.nor(1, 0), 0)
    def test_11(self): self.assertEqual(self.p.nor(1, 1), 0)


class TestXor(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.xor(0, 0), 0)
    def test_01(self): self.assertEqual(self.p.xor(0, 1), 1)
    def test_10(self): self.assertEqual(self.p.xor(1, 0), 1)
    def test_11(self): self.assertEqual(self.p.xor(1, 1), 0)


class TestXnor(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_00(self): self.assertEqual(self.p.xnor(0, 0), 1)
    def test_01(self): self.assertEqual(self.p.xnor(0, 1), 0)
    def test_10(self): self.assertEqual(self.p.xnor(1, 0), 0)
    def test_11(self): self.assertEqual(self.p.xnor(1, 1), 1)


class TestValidacion(unittest.TestCase):
    def setUp(self): self.p = Neuron()
    def test_nand_invalido(self):  self.assertRaises(ValueError, self.p.nand,  2, 0)
    def test_not_invalido(self):   self.assertRaises(ValueError, self.p.not_,  5)
    def test_and_invalido(self):   self.assertRaises(ValueError, self.p.and_,  1, -1)
    def test_or_invalido(self):    self.assertRaises(ValueError, self.p.or_,   0, 2)
    def test_nor_invalido(self):   self.assertRaises(ValueError, self.p.nor,   0, 2)
    def test_xor_invalido(self):   self.assertRaises(ValueError, self.p.xor,   3, 0)
    def test_xnor_invalido(self):  self.assertRaises(ValueError, self.p.xnor,  0, 9)
    def test_string_invalido(self):
        self.assertRaises((ValueError, TypeError), self.p.and_, "a", 1)
    def test_none_invalido(self):
        self.assertRaises((ValueError, TypeError), self.p.xnor, None, 1)
    def test_valores_validos_no_lanzan(self):
        for a, b in BITS:
            try:
                self.p.and_(a, b)
            except Exception as e:
                self.fail(f"and_({a},{b}) lanzó inesperadamente: {e}")


class TestPropiedades(unittest.TestCase):
    def setUp(self): self.p = Neuron()

    def test_conmutatividad_and(self):
        for a, b in BITS:
            self.assertEqual(self.p.and_(a, b), self.p.and_(b, a))

    def test_conmutatividad_or(self):
        for a, b in BITS:
            self.assertEqual(self.p.or_(a, b), self.p.or_(b, a))

    def test_conmutatividad_xor(self):
        for a, b in BITS:
            self.assertEqual(self.p.xor(a, b), self.p.xor(b, a))

    def test_doble_negacion(self):
        for a in (0, 1):
            self.assertEqual(self.p.not_(self.p.not_(a)), a)

    def test_xnor_es_not_xor(self):
        for a, b in BITS:
            self.assertEqual(self.p.xnor(a, b), self.p.not_(self.p.xor(a, b)))

    def test_de_morgan_and(self):
        for a, b in BITS:
            self.assertEqual(
                self.p.not_(self.p.and_(a, b)),
                self.p.or_(self.p.not_(a), self.p.not_(b))
            )

    def test_de_morgan_or(self):
        for a, b in BITS:
            self.assertEqual(
                self.p.not_(self.p.or_(a, b)),
                self.p.and_(self.p.not_(a), self.p.not_(b))
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)