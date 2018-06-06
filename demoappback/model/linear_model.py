import warnings

import numpy as np
import pyglimmpse as pg

from demoappback.model.enums import PolynomialMatrices, HypothesisType
from demoappback.model.isu_factors import IsuFactors
from demoappback.model.study_design import StudyDesign
from demoappback.utilities import kronecker_list


class LinearModel(object):
    """class describing a GLMM"""

    def __init__(self,
                 essence_design_matrix: np.matrix = None,
                 repeated_rows_in_design_matrix: float = None,
                 beta: np.matrix = None,
                 c_matrix: np.matrix = None,
                 u_matrix: np.matrix = None,
                 sigma_star: np.matrix = None,
                 theta_zero: np.matrix = None,
                 alpha: float = None,
                 **kwargs):
        """
        Parameters
        ----------
        essence_design_matrix
            the design matrix X
        covariance_matrix
            BETA, the matrix of hypothesized regression coefficients
        c_matrix
            the "between" subject contrast for pre-multiplying BETA
        u_matrix
            the "within" subject contrast for post-multiplying BETA
        sigma
            SIGMA, the hypothesized covariance matrix
        theta_zero
            the matrix of constants to be subtracted from C*BETA*U (CBU)
        """
        self.essence_design_matrix = essence_design_matrix
        self.repeated_rows_in_design_matrix = repeated_rows_in_design_matrix
        self.beta = beta
        self.c_matrix = c_matrix
        self.u_matrix = u_matrix
        self.sigma_star = sigma_star
        self.theta_zero = theta_zero
        self.alpha = alpha

        if kwargs.get('study_design'):
            self.from_study_design(kwargs['study_design'])

    def from_study_design(self, study_design: StudyDesign):
        self.essence_design_matrix = self.calculate_design_matrix(study_design.isu_factors.get_predictors())
        self.repeated_rows_in_design_matrix = self.get_rep_n_from_study_design(study_design)
        self.beta = study_design.isu_factors.marginal_means
        self.c_matrix = self.calculate_c_matrix(study_design.isu_factors)
        self.u_matrix = self.calculate_u_matrix(study_design.isu_factors)
        self.sigma_star = self.calculate_sigma_star(study_design.isu_factors)
        self.theta_zero = 0
        self.alpha = study_design.alpha

    @staticmethod
    def calculate_design_matrix(predictors):
        components = [np.identity(1)].append([np.identity(len(p.values)) for p in predictors])
        return kronecker_list(components)

    def get_rep_n_from_study_design(self, study_design):
        return study_design.isu_factors.smallest_group_size

    def calculate_c_matrix(self, predictors):
        partials = [self.calculate_partial_c_matrix(p) for p in predictors]
        c_matrix = kronecker_list(partials)
        return c_matrix

    @staticmethod
    def calculate_u_matrix(isu_factors):
        u_outcomes = np.identity(len(isu_factors.get_outcomes()))
        u_cluster = 1
        u_repeated_measures = kronecker_list([r.partia_u_matrix for r in isu_factors.get_repeated_measures()])

        u_matrix = kronecker_list([u_outcomes, u_cluster, u_repeated_measures])
        return u_matrix

    def calculate_partial_c_matrix(self, predictor):
        partial = None
        if predictor.in_hypothesis:
            if predictor.isu_factor_nature == HypothesisType.GLOBAL_TRENDS:
                partial = self.calculate_main_effect_partial_c_matrix(predictor)
            if predictor.isu_factor_nature == HypothesisType.IDENTITY:
                partial = self.calculate_identity_partial_c_matrix(predictor)
            if predictor.isu_factor_nature == HypothesisType.POLYNOMIAL:
                partial = self.calculate_polynomial_partial_c_matrix(predictor)
            if predictor.isu_factor_nature == HypothesisType.USER_DEFINED:
                partial = predictor.partial_matrix
        else:
            partial = self.calculate_average_partial_c_matrix(predictor)
        return partial

    @staticmethod
    def calculate_average_partial_c_matrix(predictor):
        no_groups = len(predictor.values)
        average_matrix = np.ones(no_groups)/no_groups
        return average_matrix

    @staticmethod
    def calculate_main_effect_partial_c_matrix(predictor):
        i = np.identity(len(predictor.values))
        v = np.ones(len(predictor.values))
        main_effect = np.concatenate(v, i, axis=0)
        return main_effect

    @staticmethod
    def calculate_polynomial_partial_c_matrix(predictor):
        values = None
        no_groups = len(predictor.values)
        if no_groups < 2:
            warnings.warn('You have less than 2 valueNames in your main effect. This is not valid.')
        elif no_groups == 2:
            values = np.matrix(PolynomialMatrices.LINEAR_POLYNOMIAL_CMATRIX)
        elif no_groups == 3:
            values = np.matrix(PolynomialMatrices.QUADRATIC_POLYNOMIAL_CMATRIX)
        elif no_groups == 4:
            values = np.matrix(PolynomialMatrices.CUBIC_POLYNOMIAL_CMATRIX)
        elif no_groups == 5:
            values = np.matrix(PolynomialMatrices.QUINTIC_POLYNOMIAL_CMATRIX)
        elif no_groups == 6:
            values = np.matrix(PolynomialMatrices.SEXTIC_POLYNOMIAL_CMATRIX)
        else:
            warnings.warn('You have more than 6 valueNames in your main effect. We don\'t currently handle this :(')
        return values

    @staticmethod
    def calculate_identity_partial_c_matrix(predictor):
        return np.identity(len(predictor.values))

    def calculate_sigma_star(self, isu_factors: IsuFactors):
        outcome_component = self.calculate_outcome_sigma_star(isu_factors)
        repeated_measure_component = self.calculate_rep_measure_sigma_star(isu_factors.get_repeated_measures())
        cluster_component = self.calculate_cluster_sigma_star(isu_factors.get_clusters())
        return kronecker_list([outcome_component, repeated_measure_component, cluster_component])

    def calculate_outcome_sigma_star(self, isu_factors):
        return (isu_factors.outcome_correlation_matrix *
                np.matrix([o.standard_deviation for o in isu_factors.get_outcomes()]))

    def calculate_rep_measure_sigma_star(self, repeated_measures):
        return kronecker_list([m.correlation_matrix for m in repeated_measures])

    def calculate_cluster_sigma_star(self, cluster):
        return len(cluster.levels)


