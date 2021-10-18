import os
import platform
import subprocess
import json

from time import perf_counter

from app.calculation_service.model.scenario_inputs import ScenarioInputs
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.api import _generate_models, _calculate_power
import pandas as pd
from time import perf_counter

def write_pdf(tex_filename):
    filename, ext = os.path.splitext(tex_filename)
    # the corresponding PDF filename
    pdf_filename = filename + '.pdf'

    # compile TeX file
    subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_filename])

    # check if PDF is successfully generated
    if not os.path.exists(pdf_filename):
        raise RuntimeError('PDF output not found')

    # open PDF with platform-specific command
    if platform.system().lower() == 'darwin':
        subprocess.run(['open', pdf_filename])
    elif platform.system().lower() == 'windows':
        os.startfile(pdf_filename)
    elif platform.system().lower() == 'linux':
        subprocess.run(['xdg-open', pdf_filename])
    else:
        raise RuntimeError('Unknown operating system "{}"'.format(platform.system()))

def write_tex_file(filename, title, introduction, description, list_inputs, timings_table, deviations_table, results_table):
    f = open(filename + ".tex", "w")
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{geometry}\n")
    f.write("\\geometry{legalpaper, landscape, margin = 0.25 in }\n")
    f.write("\\usepackage{booktabs}\n")
    f.write("\\usepackage{longtable}\n")
    f.write("\\title{" + title + "}\n")
    f.write("\\author{Alasdair Macleod}\n")
    f.write("\\begin{document}\n")
    f.write("\\maketitle")
    f.write("\n")
    f.write("\\section{Introduction}")
    f.write("\n")
    f.write(introduction)
    f.write("\n")
    f.write("\\section{Study Design}")
    f.write("\n")
    f.write(description)
    f.write("\n")
    f.write("\\subsection{Inputs}")
    f.write("\n")
    f.write("\\subsubsection{Type One Error Rates}")
    f.write("\n")
    f.write(list_inputs["_typeOneErrorRate"])
    f.write("\n")
    f.write("\\subsubsection{Means Scale Factors}")
    f.write("\n")
    f.write(list_inputs["_scaleFactor"])
    f.write("\n")
    f.write("\\subsubsection{Variance Scale Factors}")
    f.write("\n")
    f.write(list_inputs["_varianceScaleFactors"])
    f.write("\n")
    f.write("\\subsubsection{Per Group Sample Size}")
    f.write("\n")
    f.write(list_inputs["smallestGroupSize"])
    f.write("\n")
    f.write("\\subsubsection{Power Method}")
    f.write("\n")
    f.write(list_inputs["_powerMethod"])
    f.write("\n")
    f.write("\\subsubsection{Tests}")
    f.write("\n")
    f.write(list_inputs["_selectedTests"])
    f.write("\n")
    f.write("\\subsubsection{Matrices}")
    f.write("\n")
    f.write(list_inputs["_essenceX"])
    f.write("\n")
    f.write(list_inputs["_C"])
    f.write("\n")
    f.write(list_inputs["_B"])
    f.write("\n")
    f.write(list_inputs["_U"])
    f.write("\n")
    f.write(list_inputs["_Sigma_Star"])
    f.write("\n")
    f.write(list_inputs["_ThetaO"])
    f.write("\n")
    f.write("\\section{Validation Results}")
    f.write("\n")
    f.write("\\subsection{Timings}")
    f.write("\n")
    f.write(timings_table)
    f.write("\n")
    f.write("\\subsection{Summary Statistics}")
    f.write("\n")
    f.write(deviations_table)
    f.write("\n")
    f.write("\\subsection{Full Validation Results}")
    f.write("\n")
    f.write(results_table)
    f.write("\n")
    f.write("\\section{References}")
    f.write("\\end{document}")
    f.close()

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

def get_summary_results(V2_results, _df_v2results, file_path):
    length = _df_v2results.shape[0] + 3
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    return _df_v2summary


def get_print_output(_df_v2results, _df_vtest, output_name):
    _df_output = pd.merge(_df_vtest, _df_v2results, how='outer',
                          left_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'],
                          right_on=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'])
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output['deviation_sas_v3'] = abs(_df_output.Power - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.sort_values(by=['Sigma Scale', 'Beta Scale', 'Total N', 'Alpha'], ignore_index=True)
    _df_output = _df_output.round(
        {'Power': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'Power_v2': 7, 'deviation_sas_v3': 7,
         'deviation_sim_v3': 7, 'deviation_v2_v3': 7})
    _df_output.Power = _df_output.Power.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    _df_output.Power_v2 = _df_output.Power_v2.apply('{:0<9}'.format)
    _df_output.deviation_v2_v3 = _df_output.deviation_v2_v3.apply('{:0<9}'.format)
    _df_output.to_excel(output_name + '.xlsx')
    _df_output["Test"] = _df_output["Test_x"]
    _df_output["GLIMMPSE V3 Power"] = _df_output["Power"]
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_output["GLIMMPSE V2 Power (deviation)"] = _df_output["Power_v2"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time']]
    _df_print = _df_print[_df_print['Test'].notna()]
    _df_output = _df_output[_df_output['Test_x'].notna()]
    return _df_output, _df_print


def get_inputs(V3_JSON, file_path):
    p = [file_path + n for n in V3_JSON]
    with open(p[0], 'r') as f:
        data = f.read()
    inputs = json.loads(data)
    list_inputs = {}
    list_inputs["_typeOneErrorRate"] = str([i for i in inputs["_typeOneErrorRate"]])
    list_inputs["_selectedTests"] = str([i for i in inputs["_selectedTests"]])
    list_inputs["_scaleFactor"] = str([i for i in inputs["_scaleFactor"]])
    list_inputs["_varianceScaleFactors"] = str([i for i in inputs["_varianceScaleFactors"]])
    list_inputs["smallestGroupSize"] = str([i for i in inputs["_isuFactors"]["smallestGroupSize"]])
    if len(inputs["_quantiles"]) > 0:
        list_inputs["_powerMethod"] = "Conditional: Quantiles " + str([i for i in inputs["_quantiles"]])
    else:
        list_inputs["_powerMethod"] = "Unconditional"
    i = ScenarioInputs().load_from_json(data)
    scenario = StudyDesign().load_from_json(data)
    models = _generate_models(scenario, i)
    model = models[0]
    list_inputs["_essenceX"] = "Es(\\mathbf{X}) = " + str(model.essence_design_matrix.tolist())
    list_inputs["_B"] = "\\mathbf{B} = " + str(model.hypothesis_beta.tolist())
    list_inputs["_C"] = "\\mathbf{C} = " + str(model.c_matrix.tolist())
    list_inputs["_U"] = "\\mathbf{U} = " + str(model.u_matrix.tolist())
    list_inputs["_Sigma_Star"] = "$\\mathbf{\\Sigma}_{*} = " + str(model.sigma_star.tolist())
    list_inputs["_ThetaO"] = "\\mathbf{\\Theta}_{0} = " + str(model.theta_zero.tolist())
    return list_inputs


def tex_table(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)

    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a', 'nan', 'nan000000'))
    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)

    _df_output, _df_print = get_print_output(_df_v2results, _df_vtest, output_name)
    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def tex_table_test_4(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Beta Scale', 'Total N'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a', 'nan', 'nan000000'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Beta Scale', 'Total N'], ignore_index=True)

    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)

    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def get_print_output_with_concat(_df_v2results, _df_vtest, output_name):
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)
    _df_output["Power_v2"] = pd.to_numeric(_df_output["Power_v2"], downcast="float")
    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output = _df_output.round(
        {'Power_v3': 7, 'SAS_Power': 7, 'deviation_sas_v3': 7, 'Sim_Power': 7, 'Power_v2': 7, 'deviation_sas_v3': 7,
         'deviation_sim_v3': 7, 'deviation_v2_v3': 7})
    _df_output.Power_v3 = _df_output.Power_v3.apply('{:0<9}'.format)
    _df_output.SAS_Power = _df_output.SAS_Power.apply('{:0<9}'.format)
    _df_output.deviation_sas_v3 = _df_output.deviation_sas_v3.apply('{:0<9}'.format)
    _df_output.Sim_Power = _df_output.Sim_Power.apply('{:0<9}'.format)
    _df_output.deviation_sim_v3 = _df_output.deviation_sim_v3.apply('{:0<9}'.format)
    _df_output.Power_v2 = _df_output.Power_v2.apply('{:0<9}'.format)
    _df_output.deviation_v2_v3 = _df_output.deviation_v2_v3.apply('{:0<9}'.format)
    _df_output.to_excel(output_name + '.xlsx')
    _df_output["Test"] = _df_output["Test_v3"]
    _df_output["GLIMMPSE V3 Power"] = _df_output["Power_v3"]
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_output["GLIMMPSE V2 Power (deviation)"] = _df_output["Power_v2"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time_v3']]
    _df_print = _df_print[_df_print['Test'].notna()]
    _df_output = _df_output[_df_output['Test_v3'].notna()]
    _df_output["Time"] = _df_output["Time_v3"]
    return _df_output, _df_print


def tex_table_test_5(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True).sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Beta Scale'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test','Sigma Scale','Beta Scale', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    length = 143
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def tex_table_test7(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power(file_path + model) for model in V3_JSON], ignore_index=True).add_suffix("_v3")
    _df_vtest = _df_vtest.sort_values(['Test_v3', 'Total N_v3'], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power)
    _df_output['deviation_sas_v2_calc'] = abs(_df_output.Power_v2 - _df_output.SAS_Power)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power)
    _df_output['deviation_sim_v2_calc'] = abs(_df_output.Power_v2 - _df_output.Sim_Power)
    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2)

    _df_output.to_excel(output_name + '.xlsx')


    length = 24
    _df_v2summary = pd.read_csv(file_path + V2_results, skiprows=length, engine='python', sep=':')
    _df_v2summary = _df_v2summary.rename(columns={_df_v2summary.columns[0]: 'Name'})
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


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
    _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
        "deviation_sas_v3"].astype('str') + ")"
    _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
        "deviation_sim_v3"].astype('str') + ")"
    _df_print = _df_output[
        ['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time']]

    _df_print = _df_print[_df_print['Test_x'].notna()]
    return _df_print.to_latex(index=False)

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
    # _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sas_v3"].astype('str') + ")"
    # _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sim_v3"].astype('str') + ")"
    # _df_print = _df_output[
    #     ['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
    #      'Alpha', 'Time']]

    # _df_print = _df_print[_df_print['Test_x'].notna()]
    # return _df_print.to_latex(index=False)

def tex_table_by_delta(file_path, output_name, V3_JSON: [], V2_results):
    _df_vtest = pd.concat([json_power_by_delta(file_path + model) for model in V3_JSON], ignore_index=True)
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=9, engine='python', na_values=('NaN', 'n/a', ' n/a'))
    _df_output=pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output.to_excel(output_name + '.xlsx')
    # _df_output["SAS Power (deviation)"] = _df_output["SAS_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sas_v3"].astype('str') + ")"
    # _df_output["Sim Power (deviation)"] = _df_output["Sim_Power"].astype('str') + " (" + _df_output[
    #     "deviation_sim_v3"].astype('str') + ")"
    # _df_print = _df_output[
    #     ['Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'Test_x', 'Sigma Scale', 'Beta Scale', 'Total N',
    #      'Alpha', 'Time']]

    # _df_print = _df_print[_df_print['Test_x'].notna()]
    # return _df_print.to_latex(index=False)


file_path = r'v2TestResults/'

TEST_1_FILENAME = "Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences"
TEST_2_FILENAME = "Example_2_Power_results_for_a_Paired_Ttest"
TEST_3_FILENAME = "Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences"
TEST_4_FILENAME = "Example_4_Power_and_confidence_limits_for_a_univariate_model"
TEST_5_FILENAME = "Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model"
TEST_6_FILENAME = "Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model"
TEST_7_FILENAME = "Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time"

GAUSSIAN_TEST_1_FILENAME = "Gaussian_Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation"
GAUSSIAN_TEST_4_FILENAME = "Gaussian_Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_the_Davies"
GAUSSIAN_TEST_5_FILENAME = "Gaussian_Example_5_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation"
GAUSSIAN_TEST_8_FILENAME = "Gaussian_Example_8 Unconditional_power_for_the_Univariate tests using Davies algorithm"

test1_timings, test1_deviations, test1_results, test1_list_inputs = tex_table(file_path, TEST_1_FILENAME, ['Test01_V3_ConditionalTwoSampleTTest.json'], 'Example_1_Power_for_a_two_sample_ttest_for_several_error_variance_values_and_mean_differences.csv')
test2_timings, test2_deviations, test2_results, test2_list_inputs = tex_table(file_path, TEST_2_FILENAME, ['Test02_V3_ConditionalPairedTTest.json'], 'Example_2_Power_results_for_a_Paired_Ttest.csv')
test3_timings, test3_deviations, test3_results, test3_list_inputs = tex_table(file_path, TEST_3_FILENAME, ['Test03_V3_ConditionalTwoSampleTTest3DPlot.json'], 'Example_3_Power_for_a_two_sample_ttest_for_various_sample_sizes_and_mean_differences.csv')
test4_timings, test4_deviations, test4_results, test4_list_inputs = tex_table_test_4(file_path,TEST_4_FILENAME, ['Example_4_Power_and_confidence_limits_for_a_univariate_model.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part2.json', 'Example_4_Power_and_confidence_limits_for_a_univariate_model_part3.json'], 'Example_4_Power_and_confidence_limits_for_a_univariate_model.csv')
test5_timings, test5_deviations, test5_results, test5_list_inputs = tex_table_test_5(file_path, TEST_5_FILENAME, ['Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.json'], 'Example_5_Power_for_a_test_of_interaction_in_a_multivariate_model.csv')
# test6_results = tex_table_by_delta(file_path, TEST_6_FILENAME, ['Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.json'], 'Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.csv')
test7_timings, test7_deviations, test7_results, test7_list_inputs = tex_table_test7(file_path, TEST_7_FILENAME, ['Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.json'], 'Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.csv')

# gaussian_test_1 = tex_table_gaussian(file_path, GAUSSIAN_TEST_1_FILENAME, ['GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_3.json'], 'Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation.csv')
# gaussian_test_4 = tex_table_gaussian(file_path, GAUSSIAN_TEST_4_FILENAME, ['GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_2.json','GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_3.json'], 'Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_Davies_algorithm.csv')
# gaussian_test_5 = tex_table_gaussian(file_path, GAUSSIAN_TEST_5_FILENAME,['GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_3.json'], 'Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_GeisserGreenhouse_and_HuynhFeldt_tests_using_the_Satterthwaite_approximation.csv')
# gaussian_test_8 = tex_table_gaussian(file_path, GAUSSIAN_TEST_8_FILENAME, ['GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_2.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_3.json'], 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_algorithm.csv')

TEST_1_TITLE = """GLMM(F) Example 1. Power for a two sample t-test for several error variance values and mean differences"""
TEST_1_STUDY_DESIGN_DESCRIPTION ="""The study design for Example 1 is a balanced, two-group design.
We calculate power for a two-sample t-test comparing the mean responses between the two groups.
The example is based on the results in  Muller, K. E., \\& Benignus, V. A. (1992). \\emph{Neurotoxicology and teratology}, \\emph{14}(3), 211-219."""

TEST_2_TITLE = """GLMM(F) Example 2. Power results for a Paired T-test"""
TEST_2_STUDY_DESIGN_DESCRIPTION = """The study design in Example 2 is a one sample design with a pre- and post-measurement for each participant.
We calculate power for a paired t-test comparing the mean responses at the pre- and post-measurements. We
express the paired t-test as a general linear hypothesis in a multivariate linear model."""

TEST_3_TITLE="""GLMM(F) Example 3. Power for a two sample t-test for various sample sizes and mean differences"""
TEST_3_STUDY_DESIGN_DESCRIPTION="""The study design for Example 3 is a balanced, two sample design witha single response variable. We calculate
power for a two-sample t-test comparing the mean responses between the two independent groups. The example
demonstrates changes in power with di erent sample sizes and mean di erences."""

TEST_4_TITLE="""GLMM(F) Example 4. Power and con dence limits for a univariate
model"""
TEST_4_STUDY_DESIGN_DESCRIPTION="""The study design for Example 4 is a balanced two group design. We calculate power for a two-sample t-test
comparing the mean response between the groups. We calculate con dence limits for the power values. The
example is based on Figure 1 from
Taylor, D. J., & Muller, K. E. (1995). Computing Con dence Bounds for Power and Sample Size of the General
Linear Univariate Model. The American Statistician, 49(1), 43-47."""

TEST_5_TITLE="""GLMM(F) Example 5. Power for a test of interaction in a multivari-
ate model"""
TEST_5_STUDY_DESIGN_DESCRIPTION="""The study design for Example 5 is a balanced four-sample designwith three repeated measures over time.
We calculate power for a test of the group by time interaction. The unstructured covariance model is most
appropriate for the design. The example demonstrates the di erence in power depending on the choice of
statistical test when assumptions of sphericity are unlikely to hold."""

TEST_7_TITLE="""Example 7. Power for a time by treatment interaction
using orthogonal polynomial contrast for time"""
TEST_7_STUDY_DESIGN_DESCRIPTION="""The study design for Example 7 is a balanced two sample design with  ve repeated measures over time. We
calculate power for a test of the time trend by treatment interaction. The example demonstrates the use of an
orthogonal polynomial contrast for the e ect of time."""

INTRODUCTION = """The following report contains validation results for the JavaStatistics library, a component of the GLIMMPSE
software system. For more information about GLIMMPSE and related publications, please visit
http://samplesizeshop.org.
The automated validation tests shown below compare power values produced by the GLIMMPSE V3 to
published results and also to simulation. Sources for published values include POWERLIB (Johnson et al. 2007)
and a SAS IML implementation of the methods described by Glueck and Muller (2003).
Validation results are listed in Section 3 of the report. Timing results show the calculation and simulation times
for the overall experiment and the mean times per power calculation. Summary statistics show the maximum
absolute deviation between the power value calculated by GLIMMPSE V3, the JavaStatistics library and the results obtained from
SAS or via simulation. The table in Section 3.3 shows the deviation values for each individual power comparison."""

all_names = [TEST_1_FILENAME, TEST_2_FILENAME, TEST_3_FILENAME, TEST_4_FILENAME, TEST_5_FILENAME, TEST_7_FILENAME]
all_titles = [TEST_1_TITLE, TEST_2_TITLE, TEST_3_TITLE, TEST_4_TITLE, TEST_5_TITLE, TEST_7_TITLE]
all_descriptions = [TEST_1_STUDY_DESIGN_DESCRIPTION, TEST_2_STUDY_DESIGN_DESCRIPTION, TEST_3_STUDY_DESIGN_DESCRIPTION, TEST_4_STUDY_DESIGN_DESCRIPTION, TEST_5_STUDY_DESIGN_DESCRIPTION, TEST_7_STUDY_DESIGN_DESCRIPTION]
all_list_inputs = [test1_list_inputs, test2_list_inputs, test3_list_inputs, test4_list_inputs, test5_list_inputs, test7_list_inputs]
all_timings = [test1_timings, test2_timings, test3_timings, test4_timings, test5_timings, test7_timings]
all_deviations = [test1_deviations, test2_deviations, test3_deviations, test4_deviations, test5_deviations, test7_deviations]
all_results = [test1_results, test2_results, test3_results, test4_results, test5_results, test7_results]


for n in all_names:
    write_tex_file(n,
                   all_titles[all_names.index(n)],
                   INTRODUCTION,
                   all_descriptions[all_names.index(n)],
                   all_list_inputs[all_names.index(n)],
                   all_timings[all_names.index(n)],
                   all_deviations[all_names.index(n)],
                   all_results[all_names.index(n)])
    write_pdf(n)

