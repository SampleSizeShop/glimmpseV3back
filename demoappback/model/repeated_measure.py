import numpy as np

from demoappback.model.isu_factor import IsuFactor, IsuFactorType, IsuFactorNature


class RepeatedMeasure(IsuFactor):
    """
    Class describing repeated measures.
    """

    def __init__(self, name: str = None, units: str = None, type: str = None, no_repeats: int = 2, partial_u_matrix: [] = [], correlation_matrix: [] = []):
        super().__init__(name, IsuFactorNature.WITHIN, IsuFactorType.REPEATED_MEASURE)
        self.units = ''
        self.type = ''
        self.no_repeats = 0
        self.partial_u_matrix = np.matrix(partial_u_matrix)
        self.correlation_matrix = np.matrix(correlation_matrix)
