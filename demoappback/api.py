from pyglimmpse import unirep, multirep, samplesize

from demoappback import app, db
import json, random
from flask import Response, request
from flask_cors import cross_origin

from demoappback.model.enums import SolveFor, Tests
from demoappback.model.linear_model import LinearModel
from demoappback.model.study_design import StudyDesign
import numpy as np


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
    json_response = json.dumps({'message':'OK', 'status':'200', 'mimetype':'application/json'})
    return json_response


@app.route('/api/calculate', methods=['POST'])
@cross_origin()
def calculate():
    """Calculate power/samplesize from a study design"""
    data = request.data
    scenario = StudyDesign().load_from_json(data)
    model = LinearModel()
    model.from_study_design(scenario)
    results = [];
    if scenario.solve_for == SolveFor.POWER:
        for test in scenario.selected_tests:
            if test == Tests.HOTELLING_LAWLEY:
                power = multirep.hlt_two_moment_null_approximator_obrien_shieh(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                                               rank_U = np.linalg.matrix_rank(model.u_matrix),
                                                                               rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                                                               total_N=model.total_n,
                                                                               eval_HINVE = np.linalg.eigvals(model.hypothesis_sum_square*model.error_sum_square),
                                                                               alpha = model.alpha)
                results.append(dict(test=Tests.HOTELLING_LAWLEY.value, power=power.power))
            elif test == Tests.PILLAI_BARTLET:
                power = multirep.pbt_two_moment_null_approx_obrien_shieh(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                                         rank_U = np.linalg.matrix_rank(model.u_matrix),
                                                                         rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                                                         total_N=model.total_n,
                                                                         eval_HINVE = np.linalg.eigvals(model.hypothesis_sum_square*model.error_sum_square),
                                                                         alpha = model.alpha)
                results.append(dict(test=Tests.PILLAI_BARTLET.value, power=power.power))
            elif test == Tests.WILKS_LIKLIEHOOD:
                power = multirep.wlk_two_moment_null_approx_obrien_shieh(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                                         rank_U = np.linalg.matrix_rank(model.u_matrix),
                                                                         rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                                                         total_N=model.total_n,
                                                                         eval_HINVE = np.linalg.eigvals(model.hypothesis_sum_square*model.error_sum_square),
                                                                         alpha = model.alpha)
                results.append(dict(test=Tests.WILKS_LIKLIEHOOD.value, power=power.power))
            elif test == Tests.BOX_CORRECTION:
                power = unirep.box(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                   rank_U = np.linalg.matrix_rank(model.u_matrix),
                                   total_N = model.total_n,
                                   rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                   error_sum_square = model.error_sum_square,
                                   hypo_sum_square = model.hypothesis_sum_square,
                                   sigma_star = model.sigma_star,
                                   alpha = model.alpha,
                                   optional_args = scenario.optional_args)
                results.append(dict(test=Tests.BOX_CORRECTION.value, power=power.power))
            elif test == Tests.GEISSER_GREENHOUSE:
                power = unirep.geisser_greenhouse(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                  rank_U = np.linalg.matrix_rank(model.u_matrix),
                                                  total_N = model.total_n,
                                                  rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                                  error_sum_square = model.error_sum_square,
                                                  hypo_sum_square = model.hypothesis_sum_square,
                                                  sigma_star = model.sigma_star,
                                                  alpha = model.alpha,
                                                  optional_args = scenario.optional_args)
                results.append(dict(test=Tests.GEISSER_GREENHOUSE.value, power=power.power))
            elif test == Tests.HUYNH_FELDT:
                power = unirep.hyuhn_feldt(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                           rank_U = np.linalg.matrix_rank(model.u_matrix),
                                           total_N = model.total_n,
                                           rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                           error_sum_square = model.error_sum_square,
                                           hypo_sum_square = model.hypothesis_sum_square,
                                           sigma_star = model.sigma_star,
                                           alpha = model.alpha,
                                           optional_args = scenario.optional_args)
                results.append(dict(test=Tests.HUYNH_FELDT.value, power=power.power))
            elif test == Tests.UNCORRECTED:
                power = unirep.uncorrected(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                           rank_U = np.linalg.matrix_rank(model.u_matrix),
                                           total_N = model.total_n,
                                           rank_X = np.linalg.matrix_rank(model.essence_design_matrix),
                                           error_sum_square = model.error_sum_square,
                                           hypo_sum_square = model.hypothesis_sum_square,
                                           sigma_star = model.sigma_star,
                                           alpha = model.alpha,
                                           optional_args = scenario.optional_args)
                results.append(dict(test=Tests.UNCORRECTED.value, power=power.power))
        json_response = json.dumps(dict(message='OK',
                                        status=200,
                                        mimetype='application/json',
                                        results=results,
                                        model=model.to_dict()))

    else:
        size = samplesize.samplesize(test=unirep._chi_muller_muller_barton_1989,
                                    rank_C=np.linalg.matrix_rank(model.c_matrix),
                                    rank_U=np.linalg.matrix_rank(model.c_matrix),
                                    alpha=model.alpha,
                                    sigmaScale=1,
                                    sigma=model.sigma_star,
                                    betaScale=1,
                                    beta=model.hypothesis_beta,
                                    targetPower=scenario.target_power,
                                    rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                    eval_HINVE=model.hypothesis_sum_square*model.nu_e)
        json_response = json.dumps(dict(message='OK',
                                        status=200,
                                        mimetype='application/json',
                                        samplesize=size,
                                        model=model.to_dict()))

    return json_response

@app.route('/')
def hello_world():
    """Hello world!"""
    return 'Hello World!'



