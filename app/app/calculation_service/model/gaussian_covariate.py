import numpy as np


class GaussianCovariate():
    """
    Class describing a Gaussian Covariate
    """

    def __init__(self,
                 standard_deviation: float,
                 correlations: np.matrix,
                 quantiles: [],
                 exact: False,
                 **kwargs):
        """
        Class describing a Gaussian Covariate

        :param standard_deviation: the standard deviation of the gaussian covariate
        :param correlations: the corellations of the gaussian covariate to the outcomes and repeated measurements of the study
        :param kwargs:
        """
        self.standard_deviation = standard_deviation
        self.correlations = correlations
        self.quantiles = quantiles
        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        if source.get('standardDeviation'):
            self.standard_deviation = source['standardDeviation']
        if source.get('correlations'):
            self.standard_deviation = np.matrix(source['correlations']['_values']['data'])
        if source.get('quantiles'):
            self.standard_deviation =source['quantiles']
