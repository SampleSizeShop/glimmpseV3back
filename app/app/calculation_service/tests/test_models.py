import unittest
import numpy as np
import re
from app.models import Matrix


class ModelTestCase(unittest.TestCase):
    m = Matrix('M')

    def setUp(self):
        self.m = Matrix('M')

    def tearDown(self):
        pass

    def test___init__(self):
        """Should return a matrix with name M and vaules _SAMPLE"""
        expected = ('M', np.matrix(self.m._SAMPLE))
        actual = (self.m.name, self.m.matrix)

        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1].tolist(), actual[1].tolist())

    def test___str__(self):
        """Should print a statement containing name and values as a list"""
        expected = False
        if self.m.name in str(self.m) and str(self.m.matrix) in str(self.m):
            expected = True
        self.assertTrue(expected)

    def test_bmatrix(self):
        """Should return a TeX bmatrix"""
        pattern = "begin{bmatrix}.*\n.*\n.*\n.*\n.*\n.*\n.*end{bmatrix}"
        self.assertTrue(re.search(pattern, self.m.bmatrix()))


if __name__ == '__main__':
    unittest.main()
