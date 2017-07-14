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


if __name__ == '__main__':
    unittest.main()
