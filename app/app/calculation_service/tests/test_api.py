import app
import unittest


class ViewsTestCase(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.app = app.app.test_client()

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
