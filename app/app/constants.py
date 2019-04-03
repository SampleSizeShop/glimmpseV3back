from enum import Enum


class Constants(Enum):
    # errors in linear model creation
    ERR_ERROR_DEG_FREEDOM = 'Error degrees of freedom must be positive. To achieve this increase smallest group size'
    ERR_NOT_POSITIVE_DEFINITE = 'Sigma star is not positive definite.'
    ERR_NO_DIFFERENCE = 'Your hypothesis and means have been chosen such that there is no difference. As such power can be no greater than your type one error rate. Please change either your hypothesis or your means.'
