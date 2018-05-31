from demoappback.model.isu_factor import IsuFactor
from demoappback.model.enums import HypothesisType, IsuFactorType, Nature
from demoappback.utilities import list_compare


class ClusterLevel(object):
    """
    Class describing cluster levels
    """

    def __init__(self,
                 level_name: int = None,
                 no_elements: int = 1,
                 intra_class_correlation = 1,
                 **kwargs):
        self.level_name = level_name
        self.no_elements = no_elements
        self.intra_class_corellation = intra_class_correlation

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def from_dict(self, source):
        if source.get('levelName'):
            self.level_name = source['levelName']
        if source.get('noElements'):
            self.no_elements = source['noElements']
        if source.get('intraClassCorellation'):
            self.intra_class_corellation = source['intraClassCorellation']

class Cluster(IsuFactor):
    """
    Class describing clusters.
    """

    def __init__(self,
                 name: str=None,
                 values: []=None,
                 in_hypothesis: bool=False,
                 child=None,
                 partial_matrix=None,
                 levels: [] = None,
                 **kwargs):
        super().__init__(name=name,
                         nature=Nature.WITHIN,
                         factor_type=IsuFactorType.CLUSTER,
                         values=values,
                         in_hypothesis=in_hypothesis,
                         hypothesis_type=HypothesisType.GLOBAL_TRENDS,
                         child=child,
                         partial_matrix=partial_matrix)
        self.levels = levels

        if kwargs.get('source'):
            self.from_dict(kwargs['source'])

    def __eq__(self, other):
        comp = []
        for key in self.__dict__:
            if key not in other.__dict__:
                comp.append(False)
            elif key == 'levels':
                comp.append(list_compare(self.levels, other.levels))
            else:
                comp.append(self.__dict__[key] == other.__dict__[key])
        return False not in comp

    def from_dict(self, source):
        super().from_dict(source)
        if source.get('levels'):
            self.levels = [ClusterLevel(source=level) for level in source['levels']]
