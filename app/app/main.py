
from flask import Flask
from flask_pymongo import MongoClient
from flask_cors import CORS

from app.calculation_service import api

app = Flask(__name__)
app.register_blueprint(api.bp)
client = MongoClient('localhost', 27017)
db = client['test-database']
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})



@app.route('/')
def hello_world():
    """Hello world!"""
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
