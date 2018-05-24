import warnings
import functools

from pyglimmpse.constants import Constants


def check_options( function ):
    """ validates the options """

    @functools.wraps( function )
    def check_options_wrapper( **kwargs ):
        CL = None
        Option = None
        for key, value in kwargs.items():
            if key == 'CL':
                CL = value
            if key == 'Option':
                Option = value

        if CL and Option and CL.cl_type == Constants.CLTYPE_NOT_DESIRED and Option.opt_noncencl:
            raise Exception("ERROR 83: NONCENCL is not a valid option when CL not desired.")
        return function( **kwargs )
    return check_options_wrapper

def repn_positive( function ):
    """ checks what exactly abuot repn????"""

    @functools.wraps( function )
    def repn_positive_wrapper ( **kwargs ):
        Scalar = None
        Option = None
        for key, value in kwargs.items():
            if key == 'Scalar':
                Scalar = value
            if key == 'Option':
                Option = value
        # Check repn
        if Scalar and Scalar.rep_n <= Scalar.tolerance:
            raise Exception('ERROR 10: All REPN values must be > TOLERANCE > 0.')

        if Scalar and Option and Option.opt_fracrepn and Scalar.rep_n % 1 != 1:
            raise Exception('ERROR 11: All REPN values must be positive integers. To allow fractional REPN values, '
                            'specify opt_fracrepn')
        return function(**kwargs)
    return repn_positive_wrapper


def parameters_positive(function):
    """ Checks the values of various properties"""

    @functools.wraps(function)
    def parameters_positive_wrapper(**kwargs):
        Scalar = None
        for key, value in kwargs.items():
            if key == 'Scalar':
                Scalar = value

        if Scalar:
            # Check sigscal
            if Scalar.sigma_scalar <= Scalar.tolerance:
                raise Exception('ERROR 12: All SIGSCAL values must be > TOLERANCE > 0.')

            # Check alpha
            if Scalar.alpha <= Scalar.tolerance or Scalar.alpha >= 1:
                raise Exception('ERROR 13: All ALPHA values must be > TOLERANCE > 0 and < 1.')

            # Check tolerance
            if Scalar.tolerance <= 0:
                raise Exception('ERROR 17: User specified TOLERANCE <= zero.')
            if Scalar.tolerance >= 0.01:
                raise Exception('WARNING 6: User specified TOLERANCE >= 0.01. This is the value assumed to be numeric '
                                'zero and affects many calculations. Please check that this value is correct.')
            return function(**kwargs)
        return parameters_positive_wrapper


def valid_approximations(function):
    """ Ensures that we are using appropriate appriximations """

    @functools.wraps(function)
    def valid_approximation_wrapper(**kwargs):
        CalcMethod = None
        for key, value in kwargs.items():
            if key == 'CalcMethod':
                CalcMethod = value
        if CalcMethod.UnirepUncorrected == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        CalcMethod.UnirepHuynhFeldt == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        CalcMethod.UnirepHuynhFeldtChiMuller == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        CalcMethod.UnirepGeisserGreenhouse == Constants.UCDF_MULLER1989_APPROXIMATION or \
                        CalcMethod.UnirepBox == Constants.UCDF_MULLER1989_APPROXIMATION:
            warnings.warn('WARNING 7: You have chosen the Muller, Barton (1989) approximation for the UNIREP '
                          'statistic CDF. Muller, Edwards, Taylor (2004) found NO condition where their approximation '
                          'was not superior to this Muller, Barton approximation.  Suggest specifying '
                          'UCDF_MULLER2004_APPROXIMATION; '
                          'unless you are performing a backwards comparison calculation.')
        return function(**kwargs)
    return valid_approximation_wrapper


def valid_internal_pilot(function):
    """ Ensure that sigma is known if we are doing an internal pilot """

    @functools.wraps(function)
    def valid_internal_pilot_wrapper(**kwargs):
        CL = None
        for key, value in kwargs.items():
            if key == 'CL':
                CL = value
            if key == 'IP':
                IP = value
        # Check IP_PLAN and SIGTYPE
        if IP.ip_plan and CL.sigma_type:
            raise Exception('ERROR 91: SIGMA must be known when planning an internal pilot.')
        return function(**kwargs)
    return valid_internal_pilot_wrapper