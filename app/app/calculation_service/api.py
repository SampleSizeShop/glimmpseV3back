from pyglimmpse import unirep, multirep, samplesize

import json, random
from flask import Blueprint, Response, request
from flask_cors import cross_origin

from app.calculation_service.model.enums import SolveFor, Tests
from app.calculation_service.model.linear_model import LinearModel
from app.calculation_service.model.study_design import StudyDesign
import numpy as np

#from app.main import db
from app.calculation_service.model.scenario_inputs import ScenarioInputs

bp = Blueprint('pyglimmpse', __name__, url_prefix='/api')


def jsonify_tex(texString):
    data = {'texString': texString}
    json_response = json.dumps(data)
    return Response(json_response, status=200, mimetype='application/json')

#
# @bp.route('/storedtex', methods=['POST'])
# @cross_origin()
# def storedexpression():
#     """return a TeX expression from Mongo DB"""
#     cur = db.expressions.find({'name': '{0}'.format(random.randrange(1, 6, 1))})
#     expr = cur.next()['expression']
#     return jsonify_tex(expr)


@bp.route('/clientsidelog', methods=['POST'])
@cross_origin()
def client_side_log():
    """print a log recieved from the client side logger"""
    level = request.get_json()
    print(level)
    json_response = json.dumps({'message':'OK', 'status':'200', 'mimetype':'application/json'})
    return json_response


@bp.route('/calculate', methods=['POST'])
@cross_origin()
def calculate():
    """Calculate power/samplesize from a study design"""
    data = request.data
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)

    results = []
    for model in models:
        if scenario.solve_for == SolveFor.POWER:
            result = calculate_power(model, scenario)
        else:
            result = calculate_sample_size(model, scenario)
        results.append(result[0])

        if model.errors:
            message = 'Error!'
        else:
            message = 'OK'

    json_response = json.dumps(dict(message=message,
                                    status=200,
                                    mimetype='application/json',
                                    results=results,
                                    model=model.to_dict()))

    return json_response


def _generate_models(scenario: StudyDesign, inputs: []):
    """ Create a LinearModel object for each distinct set of parameters defined in the scenario"""
    models = []
    for inputSet in inputs:
            model = LinearModel()
            model.from_study_design(scenario, inputSet)
            models.append(model)
    return models


def calculate_sample_size(model, scenario):
    results = []
    if model.test == Tests.HOTELLING_LAWLEY:
        size = samplesize.samplesize(test=multirep.hlt_two_moment_null_approximator_obrien_shieh,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     eval_HINVE=model.hypothesis_sum_square * model.nu_e,
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square)
        results.append(dict(test=Tests.HOTELLING_LAWLEY.value, samplesize=size))
    elif model.test == Tests.PILLAI_BARTLET:
        size = samplesize.samplesize(test=multirep.pbt_two_moment_null_approx_obrien_shieh,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     eval_HINVE=model.hypothesis_sum_square * model.nu_e,
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square)
        results.append(dict(test=Tests.PILLAI_BARTLET.value, samplesize=size))
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        size = samplesize.samplesize(test=multirep.wlk_two_moment_null_approx_obrien_shieh,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     eval_HINVE=model.hypothesis_sum_square * model.nu_e,
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square)
        results.append(dict(test=Tests.WILKS_LIKLIEHOOD.value, samplesize=size))
    elif model.test == Tests.BOX_CORRECTION:
        size = samplesize.samplesize(test=unirep.box,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square,
                                     optional_args=scenario.optional_args)
        results.append(dict(test=Tests.BOX_CORRECTION.value, samplesize=size))
    elif model.test == Tests.GEISSER_GREENHOUSE:
        size = samplesize.samplesize(test=unirep.geisser_greenhouse,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square,
                                     optional_args=scenario.optional_args)
        results.append(dict(test=Tests.GEISSER_GREENHOUSE.value, samplesize=size))
    elif model.test == Tests.HUYNH_FELDT:
        size = samplesize.samplesize(test=unirep.hyuhn_feldt,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square,
                                     optional_args=scenario.optional_args)
        results.append(dict(test=Tests.HUYNH_FELDT.value, samplesize=size))
    elif model.test == Tests.UNCORRECTED:
        size = samplesize.samplesize(test=unirep.uncorrected,
                                     rank_C=np.linalg.matrix_rank(model.c_matrix),
                                     rank_U=np.linalg.matrix_rank(model.c_matrix),
                                     alpha=model.alpha,
                                     sigmaScale=1,
                                     sigma=model.sigma_star,
                                     betaScale=1,
                                     beta=model.hypothesis_beta,
                                     targetPower=scenario.target_power,
                                     rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                     error_sum_square=model.error_sum_square,
                                     hypothesis_sum_square=model.hypothesis_sum_square,
                                     optional_args=scenario.optional_args)
        results.append(dict(test=Tests.UNCORRECTED.value, samplesize=size))
    return results


def calculate_power(model, scenario):
    results = []
    if model.test == Tests.HOTELLING_LAWLEY:
        power = multirep.hlt_two_moment_null_approximator_obrien_shieh(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                                       rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                                       rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                                                       total_N=model.total_n,
                                                                       alpha=model.alpha,
                                                                       error_sum_square=model.error_sum_square,
                                                                       hypothesis_sum_square=model.hypothesis_sum_square)
        results.append(dict(test=Tests.HOTELLING_LAWLEY.value, power=power.power))
    elif model.test == Tests.PILLAI_BARTLET:
        power = multirep.pbt_two_moment_null_approx(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                    rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                    rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                                    total_N=model.total_n,
                                                    alpha=model.alpha,
                                                    error_sum_square=model.error_sum_square,
                                                    hypothesis_sum_square=model.hypothesis_sum_square)
        results.append(dict(test=Tests.PILLAI_BARTLET.value, power=power.power))
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        power = multirep.wlk_two_moment_null_approx(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                                    rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                    rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                                    total_N=model.total_n,
                                                    alpha=model.alpha,
                                                    error_sum_square=model.error_sum_square,
                                                    hypothesis_sum_square=model.hypothesis_sum_square)

        if power.error_message:
            model.errors.append(dict(errorname=test, errormessage=power.error_message))

        results.append(dict(test=Tests.WILKS_LIKLIEHOOD.value, power=power.power))
    elif model.test == Tests.BOX_CORRECTION:
        power = unirep.box(rank_C=np.linalg.matrix_rank(model.c_matrix),
                           rank_U=np.linalg.matrix_rank(model.u_matrix),
                           total_N=model.total_n,
                           rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                           error_sum_square=model.error_sum_square,
                           hypo_sum_square=model.hypothesis_sum_square,
                           sigma_star=model.sigma_star,
                           alpha=model.alpha,
                           optional_args=scenario.optional_args)
        results.append(dict(test=Tests.BOX_CORRECTION.value, power=power.power))
    elif model.test == Tests.GEISSER_GREENHOUSE:
        power = unirep.geisser_greenhouse(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                          rank_U=np.linalg.matrix_rank(model.u_matrix),
                                          total_N=model.total_n,
                                          rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                          error_sum_square=model.error_sum_square,
                                          hypo_sum_square=model.hypothesis_sum_square,
                                          sigma_star=model.sigma_star,
                                          alpha=model.alpha,
                                          optional_args=scenario.optional_args)
        results.append(dict(test=Tests.GEISSER_GREENHOUSE.value, power=power.power))
    elif model.test == Tests.HUYNH_FELDT:
        power = unirep.hyuhn_feldt(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                   rank_U=np.linalg.matrix_rank(model.u_matrix),
                                   total_N=model.total_n,
                                   rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                   error_sum_square=model.error_sum_square,
                                   hypo_sum_square=model.hypothesis_sum_square,
                                   sigma_star=model.sigma_star,
                                   alpha=model.alpha,
                                   optional_args=scenario.optional_args)
        results.append(dict(test=Tests.HUYNH_FELDT.value, power=power.power))
    elif model.test == Tests.UNCORRECTED:
        power = unirep.uncorrected(rank_C=np.linalg.matrix_rank(model.c_matrix),
                                   rank_U=np.linalg.matrix_rank(model.u_matrix),
                                   total_N=model.total_n,
                                   rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                   error_sum_square=model.error_sum_square,
                                   hypo_sum_square=model.hypothesis_sum_square,
                                   sigma_star=model.sigma_star,
                                   alpha=model.alpha,
                                   optional_args=scenario.optional_args)
        results.append(dict(test=Tests.UNCORRECTED.value, power=power.power))
    return results
