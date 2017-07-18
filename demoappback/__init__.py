from flask import Flask
from flask_pymongo import MongoClient
from flask_cors import CORS

app = Flask(__name__)
client = MongoClient('localhost', 27017)
db = client['test-database']
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

import demoappback.models
import demoappback.views

if __name__ == '__main__':
    app.run()
