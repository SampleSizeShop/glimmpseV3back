from demoappback import app, db
import json, random
from flask import Response
from flask_cors import cross_origin


@app.route('/cmatrix', methods=['POST'])
@cross_origin()
def cmatrix():
    """return a 5x5 cmatrix"""
    texString = "$C = \\begin{pmatrix}" \
                "c_{11} & c_{12} & c_{13} & c_{14} & c_{15}\\\\ " \
                "c_{21} & c_{22} & c_{23} & c_{24} & c_{25}\\\\" \
                "c_{31} & c_{32} & c_{33} & c_{34} & c_{35}\\\\" \
                "c_{41} & c_{42} & c_{43} & c_{44} & c_{45}\\\\" \
                "c_{51} & c_{52} & c_{53} & c_{54} & c_{55}" \
                "\\end{pmatrix}$"

    return jsonify_tex(texString)


def jsonify_tex(texString):
    data = {'texString': texString}
    json_response = json.dumps(data)
    return Response(json_response, status=200, mimetype='application/json')


@app.route('/storedtex', methods=['POST'])
@cross_origin()
def storedexpression():
    """return a TeX expression from Mongo DB"""
    cur = db.expressions.find({'name': '{0}'.format(random.randrange(1, 6, 1))})
    expr = cur.next()['expression']
    return jsonify_tex(expr)


@app.route('/')
def hello_world():
    """Hello world!"""
    return 'Hello World!'
