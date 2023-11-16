import sentry_sdk
from flask import Flask
from sentry_sdk.integrations.flask import FlaskIntegration
from flask_pymongo import MongoClient
from flask_cors import CORS
from app.calculation_service import api

sentry_sdk.init(
    dsn="https://36775e9f3a3a43fbb055870c286b4a64@o4504611303260160.ingest.sentry.io/4504611305422848",
    integrations=[
        FlaskIntegration(),
    ],

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

app = Flask(__name__)
app.register_blueprint(api.bp)
client = MongoClient('localhost', 27017)
db = client['test-database']
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
