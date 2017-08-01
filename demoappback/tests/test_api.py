from unittest.mock import patch

import demoappback
from demoappback import db
import unittest
import json
import re

from demoappback.api import storedexpression


class ViewsTestCase(unittest.TestCase):
    def setUp(self):
        demoappback.app.testing = True
        self.app = demoappback.app.test_client()

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
