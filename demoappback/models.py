import numpy as np

class Matrix(object):
    _SAMPLE = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]

    def __init__(self, name):
        self.name = name
        self.matrix = np.matrix(self._SAMPLE)

    def __str__(self):
        return """name {0} \n values: {1}""".format(self.name, str(self.matrix))

    def bmatrix(self):
        """Returns a LaTeX bmatrix

            :returns: LaTeX bmatrix as a string
        """
        if len(self.matrix.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(self.matrix).replace('[', '').replace(']', '').splitlines()
        texstring = [r'\begin{bmatrix}']
        texstring += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        texstring += [r'\end{bmatrix}']
        texstring =  '\n'.join(texstring)
        return texstring
