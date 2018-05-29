import json

import numpy as np

from demoappback.model.isu_factor import TargetEvent, SolveFor
from demoappback.model.isu_factors import IsuFactors
from demoappback.validators import check_options, repn_positive, parameters_positive, valid_approximations, valid_internal_pilot


class StudyDesign:
    """contains hypothesis and properties"""
    def __init__(self,
                 isu_factors: IsuFactors = None,
                 target_event: TargetEvent = None,
                 solve_for: SolveFor = None,
                 alpha = 0.05,
                 confidence_interval_width: int = None,
                 sample_size: int = 2,
                 target_power: float = None,
                 selected_tests: [] = None,
                 gaussian_covariate: float = None,
                 scale_factor: float = None,
                 variance_scale_fator: float = None,
                 power_curve: int = None):

        # I think these properties will probably end up as variables in power calc methods....
        ######################################
        # Calculated
        #self.df1 = 0
        #self.df2 = 0
        #self.dfh = []
        #self.dfe2 = 0
        ########################################
        #self.alphatest = 0
        #self.n2 = 0
        #self.cl_type = 0
        #self.n_est = 0
        #self.rank_est = 0
        #self.alpha_cl = 0
        #self.alpha_cu = 0
        #self.tolerance = 0.000000000000000001
        #self.omega = 0
        #self.power = Power()
        #self.exceptions = []

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
        self.variance_scale_fator = variance_scale_fator
        self.power_curve = power_curve

        # calculated
        self.essencex = np.matrix()
        self.beta = np.matrix()
        self.c_matrix = np.matrix()
        self.u_matrix = np.matrix()
        self.sigma = 0
        self.theta_zero = 0


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
        source = json.loads(json_str)

        if source['_solveFor']:
            self.solve_for = source['_solveFor']


