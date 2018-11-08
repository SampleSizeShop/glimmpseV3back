import numpy as np


class ContrastMatrix(object):
    """Class to store a custom contrast matrix"""

    def __init_(self,
                hypothesis_type: str = None,
                values: np.matrix = None):
        self.hypothesis_type = hypothesis_type
        self.values = values

    def from_dict(self, source):
        if source['_type']:
            self.hypothesis_type =  source['_type']
        if source['_values'] and source['_values']['data']:
            self.values =  np.matrix(source['_values']['data'])