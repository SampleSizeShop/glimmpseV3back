import demoappback
import unittest
import json

class DemoappbackTestCase(unittest.TestCase):
    def setUp(self):
        demoappback.app.testing = True
        self.app = demoappback.app.test_client()

    def tearDown(self):
        pass

    def test_mcsquared(self):
        """Should return {"texString": "$e=mc^2$"}"""
        expected = '$e = mc^2$'
        with demoappback.app.app_context():
            response = self.app.post('/mcsquared')
            data = json.loads(response.data)
            actual = data['texString']
            self.assertEqual(expected, actual)

    def test_cmatrix(self):
        """It should return a 5x5 matrix in TeX format"""
        expected = "$C = \\begin{pmatrix}" \
                   "c_{11} & c_{12} & c_{13} & c_{14} & c_{15}\\\\ " \
                   "c_{21} & c_{22} & c_{23} & c_{24} & c_{25}\\\\" \
                   "c_{31} & c_{32} & c_{33} & c_{34} & c_{35}\\\\" \
                   "c_{41} & c_{42} & c_{43} & c_{44} & c_{45}\\\\" \
                   "c_{51} & c_{52} & c_{53} & c_{54} & c_{55}" \
                   "\\end{pmatrix}$"
        with demoappback.app.app_context():
            response = self.app.post('/cmatrix')
            data = json.loads(response.data)
            actual = data['texString']
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
