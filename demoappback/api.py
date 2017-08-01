from demoappback import app, db
import json, random
from flask import Response, request
from flask_cors import cross_origin


def jsonify_tex(texString):
    data = {'texString': texString}
    json_response = json.dumps(data)
    return Response(json_response, status=200, mimetype='application/json')


@app.route('/api/storedtex', methods=['POST'])
@cross_origin()
def storedexpression():
    """return a TeX expression from Mongo DB"""
    cur = db.expressions.find({'name': '{0}'.format(random.randrange(1, 6, 1))})
    expr = cur.next()['expression']
    return jsonify_tex(expr)

@app.route('/api/clientsidelog', methods=['POST'])
@cross_origin()
def client_side_log():
    """print a log recieved from the client side logger"""
    level = request.get_json()
    print(level)
    return Response(None, status=200, mimetype='application/json')

@app.route('/')
def hello_world():
    """Hello world!"""
    return 'Hello World!'
