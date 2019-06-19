import math
import traceback

from pyglimmpse import unirep, multirep, samplesize

import json, random
from flask import Blueprint, Response, request
from flask_cors import cross_origin
from pyglimmpse.exceptions.glimmpse_exception import GlimmpseValidationException

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
            if model.errors:
                print(model.errors)
                result = dict(test=model.getTest(),
                              samplesize=model.print_errors(),
                              power=model.print_errors(),
                              model=model.to_dict())
            elif scenario.solve_for == SolveFor.POWER:
                result = _calculate_power(model)
            else:
                result = _calculate_sample_size(model)
        except GlimmpseValidationException as e:
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


def _calculate_sample_size(model):
    size = None
    if model.errors:
        pass
    elif model.test == Tests.HOTELLING_LAWLEY:
        test = multirep.hlt_two_moment_null_approximator_obrien_shieh
    elif model.test == Tests.PILLAI_BARTLET:
        test = multirep.pbt_two_moment_null_approx_obrien_shieh
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        test = multirep.wlk_two_moment_null_approx_obrien_shieh
    elif model.test == Tests.BOX_CORRECTION:
        test = unirep.box
    elif model.test == Tests.GEISSER_GREENHOUSE:
        test = unirep.geisser_greenhouse
    elif model.test == Tests.HUYNH_FELDT:
        test = unirep.hyuhn_feldt
    elif model.test == Tests.UNCORRECTED:
        test = unirep.uncorrected
    size, power = _samplesize(test=test, model=model)
    result = _samplesize_to_dict(model=model,
                                 size=size,
                                 power=power)
    return result


def _calculate_power(model):
    power = None
    if model.errors:
        pass
    elif model.test == Tests.HOTELLING_LAWLEY:
        test = multirep.hlt_two_moment_null_approximator_obrien_shieh
    elif model.test == Tests.PILLAI_BARTLET:
        test = multirep.pbt_two_moment_null_approx_obrien_shieh
    elif model.test == Tests.WILKS_LIKLIEHOOD:
        test = multirep.wlk_two_moment_null_approx_obrien_shieh
    elif model.test == Tests.BOX_CORRECTION:
        test = unirep.box
    elif model.test == Tests.GEISSER_GREENHOUSE:
        test = unirep.geisser_greenhouse
    elif model.test == Tests.HUYNH_FELDT:
        test = unirep.hyuhn_feldt
    elif model.test == Tests.UNCORRECTED:
        test = unirep.uncorrected
    power = _power(test=test, model=model)
    result = _power_to_dict(model=model, power=power)
    return result


def _samplesize(test, model, **kwargs):
    if model.noncentrality_distribution:
        kwargs['noncentrality_distribution'] = model.noncentrality_distribution
    if model.quantile:
        kwargs['quantile'] = model.quantile
    if model.confidence_interval:
        kwargs['confidence_interval'] = model.confidence_interval
    kwargs['tolerance'] = 1e-12
    size, power = samplesize.samplesize(test=test,
                                        rank_C=np.linalg.matrix_rank(model.c_matrix),
                                        rank_X=model.get_rank_x(),
                                        relative_group_sizes=model.groups,
                                        alpha=model.alpha,
                                        sigma_star=model.sigma_star,
                                        delta_es=model.delta,
                                        targetPower=model.target_power,
                                        starting_smallest_group_size=model.minimum_smallest_group_size,
                                        **kwargs)

    return size, power


def _samplesize_to_dict(model, size, power):
    pow = 'Not Calculated.'
    lower = None
    upper = None
    if power:
        pow = power.power
        if power.lower_bound and power.lower_bound.power:
            lower = power.lower_bound.power
        if power.upper_bound and power.upper_bound.power:
            upper = power.upper_bound.power
    return dict(test=model.test.value,
                samplesize=size,
                power=pow,
                lower_bound=lower,
                upper_bound=upper,
                model=model.to_dict())


def _power(test, model, **kwargs):
    if model.noncentrality_distribution:
        kwargs['noncentrality_distribution'] = model.noncentrality_distribution
    if model.quantile:
        kwargs['quantile'] = model.quantile
    if model.confidence_interval:
        kwargs['confidence_interval'] = model.confidence_interval

    power = test(rank_C=np.linalg.matrix_rank(model.c_matrix),
                 rank_X=model.get_rank_x(),
                 rep_N=model.smallest_group_size,
                 relative_group_sizes=model.groups,
                 alpha=model.alpha,
                 sigma_star=model.sigma_star,
                 delta_es=model.delta,
                 **kwargs)
    return power


def _power_to_dict(model, power):
    pow = 'Not Calculated.'
    lower = None
    upper = None
    if power:
        pow = power.power
        if math.isnan(pow):
            pow = -1
            model.errors.add(Constants.ERR_ERROR_DEG_FREEDOM)
        if power.lower_bound and power.lower_bound.power:
            lower = power.lower_bound.power
            if math.isnan(lower):
                lower = -1
        if power.upper_bound and power.upper_bound.power:
            upper = power.upper_bound.power
            if math.isnan(upper):
                upper = -1
    result = dict(test=model.test.value,
                  power=pow,
                  lower_bound=lower,
                  upper_bound=upper,
                  model=model.to_dict())
    return result
