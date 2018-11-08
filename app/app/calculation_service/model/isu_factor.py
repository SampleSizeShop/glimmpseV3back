from app.calculation_service.model.enums import IsuFactorType, HypothesisType, Nature
import numpy as np

class IsuFactor(object):
    """
    Parent class for indepenent sampling unit factors (ISUs).

    Outcomes
    Repeated Measures
    Clusters
    """

    def __init__(self,
                 name: str=None,
                 nature: str=None,
                 factor_type: IsuFactorType=None,
                 values: []=None, in_hypothesis: bool=False,
                 hypothesis_type: HypothesisType =None,
                 child=None,
                 partial_matrix=None):
        self.name = name
        self.nature = nature
        self.factor_type = factor_type
        self.values = values
        self.in_hypothesis = in_hypothesis
        self.hypothesis_type = hypothesis_type
        self.child = child
        self.partialMatrix = partial_matrix

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def from_dict(self, source):
        if source.get('name'):
            self.name = source['name']
        if source.get('nature'):
            self.nature = Nature(source['nature'])
        if source.get('valueNames'):
            self.values = [value for value in source['valueNames']]
        if source.get('inHypothesis'):
            self.in_hypothesis = source['inHypothesis']
        if source.get('isuFactorNature'):
            self.hypothesis_type = HypothesisType(source['isuFactorNature'])
        if source.get('child'):
            self.child = source['child']
        if source.get('partialMatrix'):
            self.partial_matrix = np.matrix(source['partialMatrix']['_values']['data'])
