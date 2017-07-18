from unittest.mock import patch

import demoappback
from demoappback import db
import unittest
import json
import re

from demoappback.views import storedexpression


class ViewsTestCase(unittest.TestCase):
    def setUp(self):
        demoappback.app.testing = True
        self.app = demoappback.app.test_client()

    def tearDown(self):
        pass

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
