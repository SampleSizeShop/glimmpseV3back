from flask import Flask
app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

import demoappback.views

if __name__ == '__main__':
    app.run()
