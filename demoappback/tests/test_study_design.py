import unittest
import numpy as np
import re

from demoappback.model.study_design import StudyDesign
from demoappback.models import Matrix


class StudyDesignTestCase(unittest.TestCase):
    m = Matrix('M')

    def setUp(self):
        self.m = Matrix('M')

    def tearDown(self):
        pass

    def test___init__(self):
        """Should return a matrix with name M and vaules _SAMPLE"""
        expected = StudyDesign(
                isu_factors=None,
                target_event=None,
                solve_for=None,
                alpha=0.05,
                confidence_interval_width=None,
                sample_size=2,
                target_power=None,
                selected_tests=None,
                gaussian_covariate= None,
                scale_factor=None,
                variance_scale_factor=None,
                power_curve=None)
        actual = StudyDesign()
        self.assertEqual(vars(expected), vars(actual))

    def test___str__(self):
        """Should print a statement containing name and values as a list"""
        expected = False
        if self.m.name in str(self.m) and str(self.m.matrix) in str(self.m):
            expected = True
        self.assertTrue(expected)

    def test_load_from_json(self):
        """Should return a TeX bmatrix"""
        pattern = "begin{bmatrix}.*\n.*\n.*\n.*\n.*\n.*\n.*end{bmatrix}"
        self.assertTrue(re.search(pattern, self.m.bmatrix()))


if __name__ == '__main__':
    unittest.main()
