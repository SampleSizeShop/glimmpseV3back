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
from app.constants import Constants

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
        try:
            if scenario.solve_for == SolveFor.POWER:
                result = _calculate_power(model, scenario)
            else:
                result = _calculate_sample_size(model, scenario)
        except Exception as e:
            print(e)
            result = dict(test=model.test.value,
                          samplesize=e.args[0],
                          power=e.args[0],
                          model=model.to_dict())
        results.append(result)

    json_response = json.dumps(dict(status=200,
                                    mimetype='application/json',
                                    results=results))

    return json_response


def _generate_models(scenario: StudyDesign, inputs: []):
    """ Create a LinearModel object for each distinct set of parameters defined in the scenario"""
    models = []
    for inputSet in inputs:
            model = LinearModel()
            model.from_study_design(scenario, inputSet)
            models.append(model)
    return models


def _calculate_sample_size(model, scenario):
    size = None
    if model.test == Tests.HOTELLING_LAWLEY:
        size, power = _multirep_samplesize(test=multirep.hlt_two_moment_null_approximator_obrien_shieh,
                                    model=model)
    elif model.test == Tests.PILLAI_BARTLET:
        size, power = _multirep_samplesize(test=multirep.pbt_two_moment_null_approx_obrien_shieh,
                                    model=model)
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        size, power = _multirep_samplesize(test=multirep.wlk_two_moment_null_approx_obrien_shieh,
                                    model=model)
    elif model.test == Tests.BOX_CORRECTION:
        size, power = _unirep_samplesize(test=unirep.box,
                                  model=model,
                                  scenario=scenario)
    elif model.test == Tests.GEISSER_GREENHOUSE:
        size, power = _unirep_samplesize(test=unirep.geisser_greenhouse,
                                  model=model,
                                  scenario=scenario)
    elif model.test == Tests.HUYNH_FELDT:
        size, power = _unirep_samplesize(test=unirep.hyuhn_feldt,
                                  model=model,
                                  scenario=scenario)
    elif model.test == Tests.UNCORRECTED:
        size, power = _unirep_samplesize(test=unirep.uncorrected,
                                  model=model,
                                  scenario=scenario)
    result = _samplesize_to_dict(model=model,
                                 size=size,
                                 power=power)
    return result


def _calculate_power(model, scenario):
    power = None
    if model.errors:
        pass
    elif model.test == Tests.HOTELLING_LAWLEY:
        power = _multirep_power(test=multirep.hlt_two_moment_null_approximator_obrien_shieh,
                                model=model)
    elif model.test == Tests.PILLAI_BARTLET:
        power = _multirep_power(test=multirep.pbt_two_moment_null_approx_obrien_shieh,
                                model=model)
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        power = _multirep_power(test=multirep.wlk_two_moment_null_approx_obrien_shieh,
                                model=model)
    elif model.test == Tests.BOX_CORRECTION:
        power = _unirep_power(test=unirep.box,
                              model=model,
                              scenario=scenario)
    elif model.test == Tests.GEISSER_GREENHOUSE:
        power = _unirep_power(test=unirep.geisser_greenhouse,
                              model=model,
                              scenario=scenario)
    elif model.test == Tests.HUYNH_FELDT:
        power = _unirep_power(test=unirep.hyuhn_feldt,
                              model=model,
                              scenario=scenario)
    elif model.test == Tests.UNCORRECTED:
        power = _unirep_power(test=unirep.uncorrected,
                              model=model,
                              scenario=scenario)
    result = _power_to_dict(model=model,
                            power=power)
    return result


def _multirep_samplesize(test, model):
    size, power = samplesize.samplesize(test=test,
                                        rank_C=np.linalg.matrix_rank(model.c_matrix),
                                        rank_U=np.linalg.matrix_rank(model.u_matrix),
                                        alpha=model.alpha,
                                        sigma_star=model.sigma_star,
                                        targetPower=model.target_power,
                                        rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                        delta=model.delta(),
                                        relative_group_sizes=model.groups,
                                        starting_smallest_group_size=model.minimum_smallest_group_size)
    return size, power


def _unirep_samplesize(test, model, scenario):
    size, power = samplesize.samplesize(test=test,
                                        rank_C=np.linalg.matrix_rank(model.c_matrix),
                                        rank_U=np.linalg.matrix_rank(model.u_matrix),
                                        alpha=model.alpha,
                                        sigma_star=model.sigma_star,
                                        targetPower=model.target_power,
                                        rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                                        delta=model.delta(),
                                        relative_group_sizes=model.groups,
                                        starting_smallest_group_size=model.minimum_smallest_group_size,
                                        optional_args=scenario.optional_args)
    return size, power


def _samplesize_to_dict(model, size, power):
    return dict(test=model.test.value,
                samplesize=size,
                power=power,
                model=model.to_dict())


def _multirep_power(test, model):
    power = test(rank_C=np.linalg.matrix_rank(model.c_matrix),
                 rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                 rep_N=model.smallest_group_size,
                 relative_group_sizes=model.groups,
                 alpha=model.alpha,
                 sigma_star=model.sigma_star,
                 delta_es=model.delta())
    return power


def _unirep_power(test, model, scenario):
    power = test(rank_C=np.linalg.matrix_rank(model.c_matrix),
                 rank_X=np.linalg.matrix_rank(model.essence_design_matrix),
                 rep_N=model.smallest_group_size,
                 relative_group_sizes=model.groups,
                 alpha=model.alpha,
                 sigma_star=model.sigma_star,
                 delta_es=model.delta(),
                 optional_args=scenario.optional_args)
    return power


def _power_to_dict(model, power):
    pow = 'Not Calculated.'
    if power:
        pow = power.power
    result = dict(test=model.test.value,
                  power=pow,
                  model=model.to_dict())
    return result
