import unittest
import numpy as np
from pyglimmpse import unirep, multirep


from app.calculation_service.model.cluster import Cluster, ClusterLevel
from app.calculation_service.model.enums import TargetEvent, SolveFor, Tests
from app.calculation_service.model.isu_factors import IsuFactors, OutcomeRepeatedMeasureStDev
from app.calculation_service.model.linear_model import LinearModel
from app.calculation_service.model.outcome import Outcome
from app.calculation_service.model.power_curve import PowerCurve, ConfidenceInterval, DataSeries
from app.calculation_service.model.predictor import Predictor
from app.calculation_service.model.repeated_measure import RepeatedMeasure
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.models import Matrix
from pyglimmpse.constants import Constants


class StudyDesignTestCase(unittest.TestCase):
    m = Matrix('M')

    def setUp(self):
        self.m = Matrix('M')

    def tearDown(self):
        pass

    def test___init__(self):
        """Should return a matrix with name M and vaules _SAMPLE"""
        expected = StudyDesign(
                isu_factors=None,
                target_event=None,
                solve_for=None,
                alpha=0.05,
                confidence_interval_width=None,
                sample_size=2,
                target_power=None,
                selected_tests=None,
                gaussian_covariate= None,
                scale_factor=None,
                variance_scale_factor=None,
                power_curve=None)
        actual = StudyDesign()
        self.assertEqual(vars(expected), vars(actual))

    def test___str__(self):
        """Should print a statement containing name and values as a list"""
        expected = False
        if self.m.name in str(self.m) and str(self.m.matrix) in str(self.m):
            expected = True
        self.assertTrue(expected)

    def test_load_from_json(self):
        """Should read the study design correctly from the model on model_2.json"""
        outcome_1 = Outcome(name='one')
        outcome_2 = Outcome(name='teo')
        rep_meas_1 = RepeatedMeasure(name='repMeas', values=[0, 1], units='time', type='Numeric', partial_u_matrix=np.matrix([[1],[-1]]), correlation_matrix=np.matrix([[1, 0],[0, 1]]))
        cluster_1 = Cluster(name='clstr', levels=[ClusterLevel(level_name='1'), ClusterLevel(level_name='2', no_elements=2)])
        predictor_1 = Predictor(name='prdctr', values=['grp1', 'grp2'])

        isu_factors = IsuFactors(variables=[outcome_1, outcome_2, rep_meas_1, cluster_1, predictor_1],
                                 smallest_group_size=2,
                                 outcome_correlation_matrix=np.matrix([[1, 0], [0, 1]]),
                                 outcome_repeated_measure_st_devs=[
                                     OutcomeRepeatedMeasureStDev(outcome='one', repeated_measure='repMeas', values=[2, 3]),
                                     OutcomeRepeatedMeasureStDev(outcome='teo', repeated_measure='repMeas', values=[4, 5])])

        expected = StudyDesign(isu_factors=isu_factors,
                               target_event=TargetEvent.REJECTION,
                               target_power=0.5,
                               solve_for=SolveFor.POWER,
                               confidence_interval_width=1,
                               sample_size=10,
                               selected_tests=[Tests.HOTELLING_LAWLEY, Tests.PILLAI_BARTLET, Tests.WILKS_LIKLIEHOOD],
                               gaussian_covariate={'variance': 1},
                               scale_factor=1,
                               variance_scale_factor=[3, 4],
                               power_curve=PowerCurve(confidence_interval=ConfidenceInterval(assumptions='Beta Fixed',
                                                                                             beta_sample_size=10),
                                                      x_axis='DesiredPower',
                                                      data_series=[DataSeries(variance_scale_factor=3)]))

        json_data = open("app/tests/model_2.json")
        data = json_data.read()
        json_data.close()
        actual = StudyDesign().load_from_json(data)
        model = LinearModel()
        model.from_study_design(actual)
        power = unirep._chi_muller_muller_barton_1989(sigma_star=model.sigma_star,
                                                      rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                      total_N=model.total_n,
                                                      rank_X=np.linalg.matrix_rank(model.essence_design_matrix))

        self.assertEqual(power, 0.99365975196663026)

    def test_load_multiple_outcomes(self):
        """Should read the study design correctly from the model on model_3.json"""
        outcome_1 = Outcome(name='one')
        outcome_2 = Outcome(name='teo')
        rep_meas_1 = RepeatedMeasure(name='repMeas', values=[0, 1], units='time', type='Numeric', partial_u_matrix=np.matrix([[1],[-1]]), correlation_matrix=np.matrix([[1, 0],[0, 1]]))
        cluster_1 = Cluster(name='clstr', levels=[ClusterLevel(level_name='1'), ClusterLevel(level_name='2', no_elements=2)])
        predictor_1 = Predictor(name='prdctr', values=['grp1', 'grp2'])

        isu_factors = IsuFactors(variables=[outcome_1, outcome_2, rep_meas_1, cluster_1, predictor_1],
                                 smallest_group_size=2,
                                 outcome_correlation_matrix=np.matrix([[1, 0], [0, 1]]),
                                 outcome_repeated_measure_st_devs=[
                                     OutcomeRepeatedMeasureStDev(outcome='one', repeated_measure='repMeas', values=[2, 3]),
                                     OutcomeRepeatedMeasureStDev(outcome='teo', repeated_measure='repMeas', values=[4, 5])])

        expected = StudyDesign(isu_factors=isu_factors,
                               target_event=TargetEvent.REJECTION,
                               target_power=0.5,
                               solve_for=SolveFor.POWER,
                               confidence_interval_width=1,
                               sample_size=10,
                               selected_tests=[Tests.HOTELLING_LAWLEY, Tests.PILLAI_BARTLET, Tests.WILKS_LIKLIEHOOD],
                               gaussian_covariate={'variance': 1},
                               scale_factor=1,
                               variance_scale_factor=[3, 4],
                               power_curve=PowerCurve(confidence_interval=ConfidenceInterval(assumptions='Beta Fixed',
                                                                                             beta_sample_size=10),
                                                      x_axis='DesiredPower',
                                                      data_series=[DataSeries(variance_scale_factor=3)]))

        json_data = open("app/tests/model_3.json")
        data = json_data.read()
        json_data.close()
        actual = StudyDesign().load_from_json(data)
        model = LinearModel()
        model.from_study_design(actual)
        power = unirep._chi_muller_muller_barton_1989(sigma_star=model.sigma_star,
                                                      rank_U=np.linalg.matrix_rank(model.u_matrix),
                                                      total_N=model.total_n,
                                                      rank_X=np.linalg.matrix_rank(model.essence_design_matrix))

        self.assertEqual(power, 1.0161495124112356)


    def test_warning_wlk_two_moment_null_approx(self):
        """Should return undefined power an error messages"""

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        deliberately_fail_tolerance = 100

        actual = multirep.wlk_two_moment_null_approx(rank_C,
                                                     rank_U,
                                                     rank_X,
                                                     total_N,
                                                     alpha,
                                                     error_sum_square,
                                                     hypothesis_sum_square,
                                                     tolerance=deliberately_fail_tolerance)

        self.assertEqual(np.isnan(actual.power), True)
        self.assertEqual(np.isnan(actual.noncentrality_parameter), True)
        self.assertEqual(actual.fmethod, Constants.FMETHOD_MISSING)
        self.assertEqual(actual.error_message,'Power is missing because because the noncentrality could not be computed.')

    def test_warning_pbt_two_moment_null_approx_obrien_shieh(self):
        """Should return undefined power an error messages"""

        rank_C = 3
        rank_U = 2
        rank_X = 4
        total_N = 20
        error_sum_square = np.matrix([[9.59999999999999000000000000, 0.000000000000000444089209850],
                                      [0.000000000000000444089209850, 9.59999999999999000000000000]])
        hypothesis_sum_square = np.matrix([[1.875, 1.08253175473054], [1.08253175473054, 0.625]])
        alpha = 0.05
        deliberately_fail_tolerance = 100

        actual = multirep.pbt_two_moment_null_approx_obrien_shieh(rank_C,
                                                     rank_U,
                                                     rank_X,
                                                     total_N,
                                                     alpha,
                                                     error_sum_square,
                                                     hypothesis_sum_square,
                                                     tolerance=deliberately_fail_tolerance)
        self.assertEqual(np.isnan(actual.power), True)
        self.assertEqual(np.isnan(actual.noncentrality_parameter), True)
        self.assertEqual(actual.fmethod, Constants.FMETHOD_MISSING)
        self.assertEqual(actual.error_message, 'Power is missing because df2 or eval_HINVE is not valid.')


if __name__ == '__main__':
    unittest.main()
