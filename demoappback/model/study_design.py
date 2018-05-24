import json

import numpy as np
from demoappback.validators import check_options, repn_positive, parameters_positive, valid_approximations, valid_internal_pilot


class StudyDesign:
    """contains hypothesis and properties"""
    def __init__(self, **kwargs):
        # I think there is an object here....
        ######################################
        # Calculated
        self.df1 = 0
        self.df2 = 0
        self.dfh = []
        self.dfe2 = 0
        #######################################
        self.alphatest = 0
        self.n2 = 0
        self.cl_type = 0
        self.n_est = 0
        self.rank_est = 0
        self.alpha_cl = 0
        self.alpha_cu = 0
        self.tolerance = 0.000000000000000001
        self.omega = 0
        self.power = Power()
        self.exceptions = []

        # FED IN
        self.target_event = None
        self.samplesize = 0
        self.solve_for = ''
        self.alpha = 0.05
        self.smallest_group
        self.gaussian_covariate = None
        self.hypothesis = None
        self.hypothesis_between = None
        self.hypothesis_between = None
        self.marginal_means  = {}

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
