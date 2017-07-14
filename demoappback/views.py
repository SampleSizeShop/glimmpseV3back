from demoappback import app
import json
from flask import Response
from flask_cors import cross_origin

@app.route('/mcsquared', methods=['POST'])
@cross_origin()
def mcsquared():
    """return a tex String"""
    data = {'texString': '$e = mc^2$'}
    json_response = json.dumps(data)
    return Response(json_response, status=200, mimetype='application/json')


@app.route('/')
def hello_world():
    """Hello world!"""
    return 'Hello World!'