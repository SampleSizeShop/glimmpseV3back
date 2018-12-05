import json
import warnings
from json import JSONEncoder

import numpy as np
from app.calculation_service import utilities
from app.calculation_service.model.enums import PolynomialMatrices, HypothesisType
from app.calculation_service.model.isu_factors import IsuFactors
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.utilities import kronecker_list


class LinearModel(object):
    """class describing a GLMM"""

    def __init__(self,
                 full_beta: bool = False,
                 essence_design_matrix: np.matrix = None,
                 repeated_rows_in_design_matrix: float = None,
                 hypothesis_beta: np.matrix = None,
                 c_matrix: np.matrix = None,
                 u_matrix: np.matrix = None,
                 sigma_star: np.matrix = None,
                 theta_zero: np.matrix = None,
                 alpha: float = None,
                 total_n: float = None,
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
        self.full_beta = full_beta
        self.essence_design_matrix = essence_design_matrix
        self.repeated_rows_in_design_matrix = repeated_rows_in_design_matrix
        self.hypothesis_beta = hypothesis_beta
        self.c_matrix = c_matrix
        self.u_matrix = u_matrix
        self.sigma_star = sigma_star
        self.theta_zero = theta_zero
        self.alpha = alpha
        self.total_n = total_n
        self.theta = None
        self.m = None
        self.error_sum_square = None
        self.hypothesis_sum_square = None
        self.nu_e = None
        self.calc_metadata()
        self.errors = []

        if kwargs.get('study_design'):
            self.from_study_design(kwargs['study_design'])

    def from_study_design(self, study_design: StudyDesign):
        self.full_beta = study_design.full_beta
        self.essence_design_matrix = self.calculate_design_matrix(study_design.isu_factors)
        self.repeated_rows_in_design_matrix = self.get_rep_n_from_study_design(study_design)
        self.hypothesis_beta = self.get_beta(study_design.isu_factors)
        self.c_matrix = self.calculate_c_matrix(study_design.isu_factors)
        self.u_matrix = self.calculate_u_matrix(study_design.isu_factors)
        self.sigma_star = self.calculate_sigma_star(study_design.isu_factors)
        self.theta_zero = study_design.isu_factors.theta0
        self.alpha = study_design.alpha
        self.total_n = self.calculate_total_n(study_design.isu_factors)
        self.calc_metadata()

    def calculate_total_n(self, isu_factors):
        smallest_group = isu_factors.smallest_group_size
        groups = self.get_groups(isu_factors)
        total_n = sum([smallest_group * g for g in groups])
        return total_n

    def get_groups(self, isu_factors):
        tables = [t.get('_table') for t in isu_factors.between_isu_relative_group_sizes]
        groups = [c.get('value') for t in tables for r in t for c in r]
        return groups

    def calc_metadata(self):
        self.theta = self.calc_theta()
        self.m = self.calc_m()
        self.nu_e = self.calc_nu_e()
        self.hypothesis_sum_square = self.calc_hypothesis_sum_square()
        self.error_sum_square = self.calc_error_sum_square()

    def get_beta(self, isu_factors):
        components = [self.get_combination_table_matrix(t) for t in isu_factors.marginal_means]
        beta = np.concatenate(tuple(components), axis=1)
        return beta

    def get_combination_table_matrix(self, table):
        rows = [row for row in table.get('_table')]
        t = [self._get_row_values(row) for row in rows]
        m = np.matrix(t)
        return m

    def _get_row_values(self, row):
        return [col.get('value') for col in row]

    def calculate_design_matrix(self, isu_factors):
        predictors = isu_factors.get_predictors()
        components = [np.matrix(np.identity(1))] + [np.matrix(np.identity(len(p.values))) for p in predictors if
                                                    p.in_hypothesis]
        kron_components = kronecker_list(components)
        groups = self.get_groups(isu_factors)
        return np.repeat(kron_components, groups, axis=0)

    def get_rep_n_from_study_design(self, study_design):
        return study_design.isu_factors.smallest_group_size

    def calculate_c_matrix(self, isu_factors):
        if isu_factors.cMatrix and isu_factors.cMatrix.hypothesis_type == HypothesisType.CUSTOM_C_MATRIX.value:
            c_matrix = isu_factors.cMatrix.values
            return c_matrix
        else:
            predictors = isu_factors.get_predictors()
            if self.full_beta:
                partials = [self.calculate_partial_c_matrix(p) for p in predictors]
            else:
                partials = [self.calculate_partial_c_matrix(p) for p in predictors if p.in_hypothesis]
            partials.append(np.matrix(np.identity(1)))
            c_matrix = kronecker_list(partials)
            return c_matrix


    def calculate_u_matrix(self, isu_factors):
        if isu_factors.uMatrix and isu_factors.uMatrix.hypothesis_type == HypothesisType.CUSTOM_U_MATRIX.value:
            u_matrix = isu_factors.uMatrix.values
            return u_matrix
        else:
            u_outcomes = np.identity(len(isu_factors.get_outcomes()))
            u_cluster = np.matrix([[1]])
            u_repeated_measures = LinearModel._get_repeated_measures_u_matrix(self, isu_factors)
            u_orth = kronecker_list([u_outcomes, u_repeated_measures, u_cluster])
            return u_orth

    def _get_repeated_measures_u_matrix(self, isu_factors):
        if self.full_beta:
            partial_u_list = [LinearModel.calculate_partial_u_matrix(r) for r in isu_factors.get_repeated_measures()]
        else:
            partial_u_list = [LinearModel.calculate_partial_u_matrix(r) for r in isu_factors.get_repeated_measures() if
                              r.in_hypothesis]
        if len(partial_u_list) == 0:
            partial_u_list = [np.matrix([[1]])]
        orth_partial_u_list = [LinearModel._get_orthonormal_u_matrix(x) for x in partial_u_list]
        orth_u_repeated_measures = kronecker_list(orth_partial_u_list)
        return orth_u_repeated_measures

    @staticmethod
    def calculate_partial_u_matrix(repeated_measure):
        if repeated_measure.in_hypothesis:
            if repeated_measure.hypothesis_type == HypothesisType.USER_DEFINED:
                partial = repeated_measure.partial_matrix
            else:
                partial = repeated_measure.partial_u_matrix
        else:
            partial = LinearModel.calculate_average_partial_u_matrix(repeated_measure)
        return partial

    @staticmethod
    def _get_orthonormal_u_matrix(u_matrix):
        u_orth, t_decomp = np.linalg.qr(u_matrix)
        return u_orth

    @staticmethod
    def calculate_average_partial_u_matrix(repeated_measure):
        no_rep = len(repeated_measure.values)
        average_matrix = np.matrix(np.ones(no_rep) / no_rep)
        return average_matrix.T

    def calculate_partial_c_matrix(self, predictor):
        partial = None
        if predictor.in_hypothesis:
            if predictor.hypothesis_type == HypothesisType.GLOBAL_TRENDS:
                partial = self.calculate_main_effect_partial_c_matrix(predictor)
            if predictor.hypothesis_type == HypothesisType.IDENTITY:
                partial = self.calculate_identity_partial_c_matrix(predictor)
            if predictor.hypothesis_type == HypothesisType.POLYNOMIAL:
                partial = self.calculate_polynomial_partial_c_matrix(predictor)
            if predictor.hypothesis_type == HypothesisType.USER_DEFINED:
                partial = predictor.partial_matrix
        else:
            partial = self.calculate_average_partial_c_matrix(predictor)
        return partial

    @staticmethod
    def calculate_average_partial_c_matrix(predictor):
        no_groups = len(predictor.values)
        average_matrix = np.ones(no_groups) / no_groups
        return average_matrix

    @staticmethod
    def calculate_main_effect_partial_c_matrix(predictor):
        i = np.identity(len(predictor.values) - 1) * -1
        v = np.matrix([np.ones(len(predictor.values) - 1)])
        main_effect = np.transpose(np.concatenate((v, i), axis=0))
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
        if len(isu_factors.get_repeated_measures()) > 0:
            repeated_measure_component = self.calculate_rep_measure_sigma_star(isu_factors)
        else:
            repeated_measure_component = np.matrix([[1]])
        if len(isu_factors.get_clusters()) > 0:
            cluster_component = self.calculate_cluster_sigma_star(isu_factors.get_clusters()[0])
        else:
            cluster_component = np.matrix([[1]])
        return kronecker_list([outcome_component, repeated_measure_component, cluster_component])

    def calculate_outcome_sigma_star(self, isu_factors):
        outcomes = isu_factors.get_outcomes()
        standard_deviations = np.identity(len(outcomes)) * [o.standard_deviation for o in outcomes]
        sigma_star_outcomes = standard_deviations * isu_factors.outcome_correlation_matrix * standard_deviations
        return sigma_star_outcomes

    def calculate_rep_measure_sigma_star(self, isu_factors):
        if self.full_beta:
            repeated_measures = [measure for measure in isu_factors.get_repeated_measures()]
        else:
            repeated_measures = [measure for measure in isu_factors.get_repeated_measures() if measure.in_hypothesis]
        if len(repeated_measures) == 0:
            return np.matrix([[1]])
        else:
            sigma_star_rep_measure_components = [
                self.calculate_rep_measure_component(measure) for measure in repeated_measures
            ]
            sigma_star_rep_measures = kronecker_list(sigma_star_rep_measure_components)
            return sigma_star_rep_measures

    def calculate_rep_measure_component(self, repeated_measure):
        st = np.diag(repeated_measure.standard_deviations)
        sigma_r = st * repeated_measure.correlation_matrix * st
        u_orth = LinearModel._get_orthonormal_u_matrix(LinearModel.calculate_partial_u_matrix(repeated_measure))
        component = np.transpose(u_orth) * sigma_r * u_orth
        return component

    def calculate_cluster_sigma_star(self, cluster):
        components = [
            (1 + (level.no_elements - 1) * level.intra_class_correlation) / level.no_elements
            for level in cluster.levels
        ]
        cluster_sigma_star = 1
        for c in components:
            cluster_sigma_star = cluster_sigma_star * c
        return cluster_sigma_star

    def calc_theta(self):
        if self.c_matrix is None or self.hypothesis_beta is None or self.u_matrix is None:
            return None
        return self.c_matrix * self.hypothesis_beta * self.u_matrix

    def calc_m(self):
        if self.c_matrix is None or self.essence_design_matrix is None:
            return None
        return (self.c_matrix *
                np.linalg.inv((np.transpose(self.essence_design_matrix) * self.essence_design_matrix))
                * np.transpose(self.c_matrix))

    def calc_nu_e(self):
        if self.total_n is None or self.essence_design_matrix is None:
            return None
        nu_e = self.total_n - np.linalg.matrix_rank(self.essence_design_matrix)
        return int(nu_e)

    def calc_error_sum_square(self):
        if self.nu_e is None or self.sigma_star is None:
            return None
        return self.nu_e * self.sigma_star

    def calc_hypothesis_sum_square(self):
        if self.theta is None or self.theta_zero is None or self.m is None or np.linalg.det(self.m) == 0:
            return None
        t = (self.theta - self.theta_zero)
        return self.repeated_rows_in_design_matrix * np.transpose(t) * np.linalg.inv(self.m) * t

    def to_dict(self):
        ret = dict(essence_design_matrix=utilities.serialise_matrix(self.essence_design_matrix),
                   repeated_rows_in_design_matrix=self.repeated_rows_in_design_matrix,
                   hypothesis_beta=utilities.serialise_matrix(self.hypothesis_beta),
                   c_matrix=utilities.serialise_matrix(self.c_matrix),
                   u_matrix=utilities.serialise_matrix(self.u_matrix),
                   sigma_star=utilities.serialise_matrix(self.sigma_star),
                   theta_zero=utilities.serialise_matrix(self.theta_zero),
                   alpha=self.alpha,
                   total_n=self.total_n,
                   theta=utilities.serialise_matrix(self.theta),
                   m=utilities.serialise_matrix(self.m),
                   nu_e=self.nu_e,
                   hypothesis_sum_square=utilities.serialise_matrix(self.hypothesis_sum_square),
                   error_sum_square=utilities.serialise_matrix(self.error_sum_square),
                   errors=self.errors)
        return ret

    def serialize(self):
        return json.dumps(self, cls=LinearModelEncoder)


class LinearModelEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, LinearModel):
            return obj.to_dict()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
