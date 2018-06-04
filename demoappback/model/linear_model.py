import numpy as np

from demoappback.model.study_design import StudyDesign
from demoappback.utilities import kronecker_list


class LinearModel(object):
    """class describing a GLMM"""

    def __init__(self,
                 essence_design_matrix: np.matrix = None,
                 beta: np.matrix = None,
                 c_matrix: np.matrix = None,
                 u_matrix: np.matrix = None,
                 sigma_star: np.matrix = None,
                 theta_zero: np.matrix = None,
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
        self.design_matrix = essence_design_matrix
        self.beta = beta
        self.c_matrix = c_matrix
        self.u_matrix = u_matrix
        self.sigma_star = sigma_star
        self.theta_zero = theta_zero

        if kwargs.get('study_design'):
            self.from_study_design(kwargs['study_design'])

    def from_study_design(self, study_design: StudyDesign):
        self.essence_design_matrix = self.calculate_design_matrix(study_design.isu_factors.get_predictors())
        self.repeated_rows_in_design_matrix = self.get_rep_n_from_study_design(study_design)
        self.beta = study_design.isu_factors.marginal_means
        self.c_matrix = self.calculate_c_matrix(study_design.isu_factors)
        self.u_matrix = self.calculate_u_matrix(study_design.isu_factors)
        self.theta_zero = 0
        self.alpha = study_design.alpha

    def calculate_design_matrix(self, predictors):
        components = [np.identity(1)].append([np.identity(len(p.values)) for p in predictors])
        return kronecker_list(components)

    def get_rep_n_from_study_design(self, study_design):
        pass

    def calculate_c_matrix(self, isu_factors):
        c = np.identity(1)
        return c

    def calculate_u_matrix(self, isu_factors):
        u_outcomes = np.identity(len(isu_factors.get_outcomes()))
        u_cluster = 1
        u_repeated_measures = kronecker_list([r.partia_u_matrix for r in isu_factors.get_repeated_measures()])

        u_matrix = kronecker_list([u_outcomes, u_cluster, u_repeated_measures])
        return u_matrix
