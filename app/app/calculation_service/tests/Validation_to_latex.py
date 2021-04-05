from time import perf_counter

from app.calculation_service.model.scenario_inputs import ScenarioInputs
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.api import _generate_models, _calculate_power
import pandas as pd
from time import perf_counter

pd.set_option('precision', 7)


def json_power(json_path):
    with open(json_path, 'r') as f:
        data = f.read()
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)
    results = []
    for m in models:
        t1 = perf_counter()
        result = _calculate_power(m)
        t2 = perf_counter()
        outdata = {'Power': result['power'],
                   'Test': result['test'],
                   'Sigma Scale': result['model']['variance_scale_factor'],
                   'Beta Scale': result['model']['means_scale_factor'],
                   'Total N': result['model']['total_n'],
                   'Alpha': result['model']['alpha'],
                   'Time': t2-t1}
        results.append(outdata)

    return pd.DataFrame(results)

def json_power_with_confidence_intervals(json_path):
    with open(json_path, 'r') as f:
        data = f.read()
    inputs = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, inputs)
    results = []
    for m in models:
        t1 = perf_counter()
        result = _calculate_power(m)
        t2 = perf_counter()
        outdata = {'Power': result['power'],
                   'Lower bound v3': result['lower_bound'],
                   'Upper bound v3': result['upper_bound'],
                   'Test': result['test'],
                   'Sigma Scale': result['model']['variance_scale_factor'],
                   'Beta Scale': result['model']['means_scale_factor'],
                   'Total N': result['model']['total_n'],
                   'Alpha': result['model']['alpha'],
                   'Time': t2-t1}
        results.append(outdata)

    return pd.DataFrame(results)

def json_power_by_delta(json_path):
    deltas = [x * 0.0008 for x in range(1, 251, 1)]
    top = [x for x in range(0, 5, 1)]
    bottom = [x for x in range(5, 10, 1)]

    with open(json_path, 'r') as f:
        data = f.read()
    results = []

    for d in deltas:

        inputs = ScenarioInputs().load_from_json(data)
        scenario = StudyDesign().load_from_json(data)


        for row in top:
            cell = scenario.isu_factors.marginal_means[0].get('_table')[row][2]
            cell['value'] = cell['value'] + d
        for row in bottom:
            cell = scenario.isu_factors.marginal_means[0].get('_table')[row][2]
            cell['value'] = cell['value'] - d
        models = _generate_models(scenario, inputs)
        for m in models:
            m.errors = []
            t1 = perf_counter()
            result = _calculate_power(m)
            t2 = perf_counter()
            outdata = {'Power': result['power'],
                       'Test': result['test'],
                       'Sigma Scale': result['model']['variance_scale_factor'],
                       'Beta Scale': result['model']['means_scale_factor'],
                       'Total N': result['model']['total_n'],
                       'Alpha': result['model']['alpha'],
                        'Time': t2-t1}
            results.append(outdata)

    return pd.DataFrame(results)



def tex_table(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a','nan','nan000000'))
    _df_output = pd.merge(_df_vtest, _df_v2results, how='outer',
                          left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'],
                          right_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.sort_values(by=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'], ignore_index=True)
    _df_output = _df_output.round(
        {'Power': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'deviation_sim_v3': 7})
    _df_output.Power = _df_output.Power.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)

    _df_output.to_excel(output_name + '.xlsx')
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time']]

    _df_print[_df_print['Test_x'].notna()]
    return _df_print.to_latex(index=False)

def tex_table_test_4(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Beta Scale', 'Total N'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Beta Scale', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output.to_excel(output_name + '.xlsx')

    # _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    # _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)

    _df_output.to_excel(output_name + '.xlsx')
def tex_table_test_5(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)
    # _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    # _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)

    _df_output.to_excel(output_name + '.xlsx')

def tex_table_test7(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_vtest = _df_vtest.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power)
    _df_output['deviation_sas_v2_calc'] = abs(_df_output.Power_v2 - _df_output.SAS_Power)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power)
    _df_output['deviation_sim_v2_calc'] = abs(_df_output.Power_v2 - _df_output.Sim_Power)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2)

    _df_output.to_excel(output_name + '.xlsx')

def tex_table_test_9(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_vtest = _df_vtest.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power)
    _df_output['deviation_sas_v2_calc'] = abs(_df_output.Power_v2 - _df_output.SAS_Power)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power)
    _df_output['deviation_sim_v2_calc'] = abs(_df_output.Power_v2 - _df_output.Sim_Power)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2)

    _df_output.to_excel(output_name + '.xlsx')

def tex_table_gaussian(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON],
                          ignore_index=True).sort_values(['Test', 'Total N', 'Power'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Test'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Total N'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N', 'Power_v2'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)


    _df_output.to_excel(output_name + '.xlsx')


def tex_table_by_delta(file_path, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power_by_delta(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    # _df_output = pd.merge(_df_vtest, _df_v2results, how='outer',
    #                       left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'],
    #                       right_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
    _df_output=pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output.to_excel('Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.xlsx')
    # _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    # _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    # _df_output = _df_output.sort_values(by=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
    # _df_output = _df_output.round(
    #     {'Power': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'deviation_sim_v3': 7})
    # _df_output.Power = _df_output.Power.apply('{:0<9}'.format)
    # _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    # _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    # _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    # _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    #
    #
    # _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sas_v3"].astype('str') + ")"
    # _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sim_v3"].astype('str') + ")"
    # _df_print = _df_output[
    #     ['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
    #      'Alpha']]
    #
    # _df_print = _df_print[_df_print['Test_x'].notna()]
    # return _df_print.to_latex(index=False)


# file_path = r'C:\Users\liqian\Dropbox (UFL)\Project_QL\2017_GLIMMPSE_V3\PyGLIMMPSE\TEXT\\'
file_path = r'v2TestResults/'




# test1 = tex_table(file_path, 'Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences', ['Test01_V3_ConditionalTwoSampleTTest.json'], 'Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences.csv')
# test2 = tex_table(file_path, 'Example_2_Power_results_for_a_Paired_Ttest', ['Test02_V3_ConditionalPairedTTest.json'], 'Example_2_Power_results_for_a_Paired_Ttest.csv')
# test3 = tex_table(file_path, 'Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences', ['Test03_V3_ConditionalTwoSampleTTest3DPlot.json'], 'Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences.csv')
# test4 = tex_table_test_4(file_path, 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part2', ['Example_4_Power_and_confidence_limits_for_a_univariate_model.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part2.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part3.json'], 'Example_4_Power_and_confidence_limits_for_a_univariate_model.csv')
# test5 = tex_table_test_5(file_path, 'Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model', ['Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.json'], 'Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.csv')
# test6 = tex_table_by_delta(file_path, ['Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.json'], 'Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.csv')
# test7 = tex_table_test7(file_path, 'Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time', ['Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.json'], 'Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.csv')

gaussian_test_1 = tex_table_gaussian(file_path, 'Gaussian_Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation', ['GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_3.json'], 'Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation.csv')
gaussian_test_1 = tex_table_gaussian(file_path, 'Gaussian_Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_the_Davies Algorithm', ['GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_2.json','GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_3.json'], 'Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_Davies_algorithm.csv')
gaussian_test_1 = tex_table_gaussian(file_path, 'Gaussian_Example_5_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation',['GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_3.json'], 'Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_GeisserGreenhouse_and_HuynhFeldt_tests_using_the_Satterthwaite_approximation.csv')
gaussian_test_1 = tex_table_gaussian(file_path, 'Gaussian_Example_8 Unconditional_power_for_the_Univariate tests using Davies algorithm', ['GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_2.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_3.json'], 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_algorithm.csv')
# print(test3)
# print(test4)
# print(test5)
# print(test6)
# print(test7)