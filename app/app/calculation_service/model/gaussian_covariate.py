class GaussianCovariate(object):
    """
    Class describing a Gaussian Covariate
    """

    def __init__(self,
                 standard_deviation: float = None,
                 exact: bool = False,
                 **kwargs):
        """
        Class describing a Gaussian Covariate

        :param standard_deviation: the standard deviation of the gaussian covariate
        :param correlations: the corellations of the gaussian covariate to the outcomes and repeated measurements of the study
        :param kwargs:
        """
        self.standard_deviation = standard_deviation
        self.exact = exact
        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def from_dict(self, source):
        if source.get('standard_deviation'):
            self.standard_deviation = source['standard_deviation']
