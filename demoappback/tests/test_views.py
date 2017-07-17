import demoappback
import unittest
import json
import re


class ViewsTestCase(unittest.TestCase):
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
        pattern = '.*\\\\begin{pmatrix}.*\\\\end{pmatrix}'
        with demoappback.app.app_context():
            response = self.app.post('/cmatrix')
            data = json.loads(response.data)
            actual = data['texString']
            self.assertTrue(re.search(pattern, actual))


if __name__ == '__main__':
    unittest.main()
