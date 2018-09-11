import json
from json import JSONDecoder
from pyglimmpse.constants import Constants

from app.calculation_service.model.enums import TargetEvent, SolveFor, Nature, OptionalArgs, Tests
from app.calculation_service.model.isu_factors import IsuFactors
from app.calculation_service.model.power_curve import PowerCurve
from app.calculation_service.validators import check_options, repn_positive, parameters_positive, valid_approximations, valid_internal_pilot


class StudyDesign:
    """contains hypothesis and properties"""
    def __init__(self,
                 isu_factors: IsuFactors = None,
                 target_event: TargetEvent = None,
                 solve_for: SolveFor = None,
                 alpha: float = 0.05,
                 confidence_interval_width: int = None,
                 sample_size: int = 2,
                 target_power: float = None,
                 selected_tests: [] = None,
                 gaussian_covariate: float = None,
                 scale_factor: float = None,
                 variance_scale_factor: [] = None,
                 power_curve: int = None):

        # fed in
        self.isu_factors = isu_factors
        self.target_event = target_event
        self.solve_for = solve_for
        self.alpha = alpha
        self.confidence_interval_width = confidence_interval_width
        self.target_power = target_power
        self.sample_size = sample_size
        self.selected_tests = selected_tests
        self.gaussian_covariate = gaussian_covariate
        self.scale_factor = scale_factor
        self.variance_scale_factor = variance_scale_factor
        self.power_curve = power_curve

    def __eq__(self, other):
        comp = []
        for key in self.__dict__:
            if key not in other.__dict__:
                comp.append(False)
            elif key == 'isu_factors':
                comp.append(self.isu_factors.__eq__(other.isu_factors))
            elif key == 'power_curve':
                comp.append(True)
            else:
                comp.append(self.__dict__[key] == other.__dict__[key])
        return False not in comp

    @check_options
    @repn_positive
    @parameters_positive
    @valid_approximations
    @valid_internal_pilot
    def __pre_calc_validation(self):
        """Runs pre-calculation validation checks. Throws exceptions if any fail. Perhaps this should live in the validators module???"""
        pass

    def validate_design(self):
        """ Valudates the study design. returns True is valid. Returns False and stores exceptions on object if invalid. """
        self.exceptions = []
        try:
            self.__pre_calc_validation()
        except Exception:
            self.exceptions.push(Exception)
        if len(self.exceptions) > 0:
            return False
        else:
            return True

    def load_from_json(self, json_str: str):
        return json.loads(json_str, cls=StudyDesignDecoder)

    def calculate_c_matrix(self):
        """Calculate the C Matrix from the hypothesis"""
        partials = [p for p in self.isu_factors.get_hypothesis() if p.nature == Nature.BETWEEN]
        averages = [p for p in self.isu_factors.variables if p.nature == Nature.BETWEEN and not p.in_hypothesis]


class StudyDesignDecoder(JSONDecoder):
    def default_optional_args(self):
        args = {
            OptionalArgs.APPROXIMATION.value: Constants.UN,
            OptionalArgs.EPSILON_ESTIMATOR.value: Constants.UCDF_MULLER2004_APPROXIMATION,
            OptionalArgs.UNIREPMETHOD.value: Constants.SIGMA_KNOWN,
            OptionalArgs.N_EST.value: 33,
            OptionalArgs.RANK_EST.value: 1,
            OptionalArgs.ALPHA_CL.value: 0.025,
            OptionalArgs.ALPHA_CU.value: 0.025,
            OptionalArgs.N_IP.value: 33,
            OptionalArgs.RANK_IP.value: 1,
            OptionalArgs.TOLERANCE.value: 1e-10}
        return args

    def decode(self, s: str) -> StudyDesign:
        study_design = StudyDesign()
        d = json.loads(s)
        if d.get('_isuFactors'):
            study_design.isu_factors = IsuFactors(source=d['_isuFactors'])
        if d.get('_targetEvent'):
            study_design.target_event = TargetEvent(d['_targetEvent'])
        if d.get('_solveFor'):
            study_design.solve_for = SolveFor(d['_solveFor'])
        if d.get('_typeOneErrorRate'):
            study_design.alpha = d['_typeOneErrorRate']
        if d.get('_power'):
            study_design.target_power = d['_power']
        if d.get('_ciwidth'):
            study_design.confidence_interval_width = d['_ciwidth']
        if d.get('_selectedTests'):
            study_design.selected_tests = [Tests(t) for t in d['_selectedTests']]
        if d.get('_gaussianCovariate'):
            study_design.gaussian_covariate = d['_gaussianCovariate']
        if d.get('_scaleFactor'):
            study_design.scale_factor = d['_scaleFactor']
        if d.get('_varianceScaleFactors'):
            study_design.variance_scale_factor = d['_varianceScaleFactors']
        if d.get('_powerCurve'):
            study_design.power_curve = PowerCurve(source=d['_powerCurve'])
        study_design.optional_args = self.default_optional_args()

        return study_design
