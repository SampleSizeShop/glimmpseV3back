import numpy as np
from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature
from demoappback.utilities import list_compare


class RepeatedMeasure(IsuFactor):
    """
    Class describing repeated measures.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 units: str=None,
                 type: str=None,
                 no_repeats: int=2,
                 partial_u_matrix: np.matrix= None,
                 correlation_matrix: np.matrix = None,
                 standard_deviations: [] = None,
                 **kwargs):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.REPEATED_MEASURE,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.units = units
        self.type = type
        self.no_repeats = no_repeats
        self.partial_u_matrix = np.matrix(partial_u_matrix)
        self.correlation_matrix = np.matrix(correlation_matrix)
        self.standard_deviations = np.matrix(standard_deviations)

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def __eq__(self, other):
        comp = []
        for key in self.__dict__:
            if key not in other.__dict__:
                comp.append(False)
            elif key == 'values':
                comp.append(list_compare(self.values, other.values))
            elif key == 'partial_u_matrix':
                comp.append(np.array_equal(self.partial_u_matrix.data, other.partial_u_matrix.data))
            elif key == 'correlation_matrix':
                comp.append(np.array_equal(self.correlation_matrix.data, other.correlation_matrix.data))
            else:
                comp.append(self.__dict__[key] == other.__dict__[key])
        return False not in comp

    def from_dict(self, source):
        super().from_dict(source)
        if source.get('units'):
            self.units = source['units']
        if source.get('type'):
            self.type = source['type']
        if source.get('_noRepeats'):
            self.no_repeats = source['_noRepeats']
        if (source.get('partialUMatrix')
                and source['partialUMatrix'].get('_values')
                and source['partialUMatrix']['_values'].get('data')):
            self.partial_u_matrix = np.matrix(source['partialUMatrix']['_values']['data'])
        if (source.get('correlationMatrix')
                and source['correlationMatrix'].get('_values')
                and source['correlationMatrix']['_values'].get('data')):
            self.correlation_matrix = np.matrix(source['correlationMatrix']['_values']['data'])
        if source.get('standard_deviations'):
            self.standard_deviations = source['standard_deviations']

