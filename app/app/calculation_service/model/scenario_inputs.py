import json
from json import JSONDecoder

from app.calculation_service.model.enums import Tests, SolveFor
from app.calculation_service.model.isu_factors import IsuFactors


class ScenarioInputs:
    """Class used to hold specific input values for a model."""
    def __init__(self,
                 alpha: float = None,
                 target_power: float = None,
                 smallest_group_size: float = None,
                 scale_factor: float = 1,
                 test: Tests = None,
                 variance_scale_factor: float = 1
                 ):
        self.alpha = alpha
        if target_power:
            self.target_power = target_power
        else:
            self.target_power = None
        self.smallest_group_size = smallest_group_size
        if scale_factor:
            self.scale_factor = scale_factor
        else:
            self.scale_factor = 1
        if variance_scale_factor:
            self.variance_scale_factor = variance_scale_factor
        else:
            self.variance_scale_factor = 1
        self.test = test

    def load_from_json(self, json_str: str):
        return json.loads(json_str, cls=ScenarioInputsDecoder)


class ScenarioInputsDecoder(JSONDecoder):
    def decode(self, s: str) -> []:
        inputs = []
        alpha = [],
        target_power = [],
        tests = []
        smallest_group_size = [],
        scale_factor = [],
        variance_scale_factor = []
        solve_for = SolveFor.POWER

        d = json.loads(s)
        if d.get('_solveFor'):
            solve_for = SolveFor(d['_solveFor'])
        if d.get('_isuFactors'):
            isu_factors = IsuFactors(source=d['_isuFactors'])
            smallest_group_size = isu_factors.smallest_group_size
        if d.get('_power'):
            target_power = [val for val in d['_power']]
        if d.get('_typeOneErrorRate'):
            alpha = [val for val in d['_typeOneErrorRate']]
        if d.get('_selectedTests'):
            tests = [Tests(t) for t in d['_selectedTests']]
        if d.get('_scaleFactor'):
            scale_factor = [val for val in d['_scaleFactor']]
        if d.get('_varianceScaleFactors'):
            variance_scale_factor = [val for val in d['_varianceScaleFactors']]

        if solve_for == SolveFor.POWER:
            for a in alpha:
                    for g in smallest_group_size:
                        for s in scale_factor:
                            for t in tests:
                                for v in variance_scale_factor:
                                    i = ScenarioInputs(a, None, g, s, t, v)
                                    inputs.append(i)
        else:
            for a in alpha:
                for p in target_power:
                    for g in smallest_group_size:
                        for s in scale_factor:
                            for t in tests:
                                for v in variance_scale_factor:
                                    i = ScenarioInputs(a, p, g, s, t, v)
                                    inputs.append(i)

        return inputs
