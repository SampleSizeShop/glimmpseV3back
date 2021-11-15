import os
import platform
import subprocess
import json

from time import perf_counter

from app.calculation_service.model.scenario_inputs import ScenarioInputs
from app.calculation_service.model.study_design import StudyDesign
from app.calculation_service.api import _generate_models, _calculate_power
import pandas as pd
import numpy as np
from time import perf_counter
from typing import List

def sstar_desc(is_gaussian: bool):
    sstar_desc = '''$\\mathbf{\\Sigma}_{*} = (\\mathbf{U}\'_{o} \\mathbf{\\Sigma}_o \\mathbf{U}_o)
                            \\otimes (\\mathbf{U}_r\' \\mathbf{\\Sigma}_r \\mathbf{U}_r)
                            \\otimes (\\mathbf{U}_c\' \\mathbf{\\Sigma}_c \\mathbf{U}_c)'''

    if is_gaussian:
        sstar_desc = sstar_desc + ' - \\mathbf{U}\'\\mathbf{\\Sigma}_{yg}\\sigma_{g}^{-2}\\mathbf{\\Sigma}_{yg}\'\\mathbf{U}'
    sstar_desc += ' = '
    return sstar_desc

def array_to_matrix(arr: List[List[float]]):
    arr = np.asarray(arr)
    latex = '\\begin{bmatrix} \n'
    for row in arr:
        for i, col in enumerate(row):
            latex = latex + ' ' + str(col) + row_end(i, row)
    latex = latex + '\\end{bmatrix}'
    return latex

def row_end(i, row):
    if i == len(row) - 1:
        return ' \\\\ \n'
    return ' & '

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

def write_tex_file(is_gaussian, filename, title, introduction, description, list_inputs, timings_table, deviations_table, results_table):
    f = open(filename + ".tex", "w")
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{geometry}\n")
    f.write("\\usepackage{amsmath}\n")
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
    f.write("\n")
    f.write(list_inputs["_C"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_B"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_U"])
    f.write("\n")
    f.write("\n")
    f.write(sstar_desc(is_gaussian))
    f.write("\n")
    f.write("\n")
    f.write(array_to_matrix(list_inputs["_Sigma_Star"]))
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_ThetaO"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_Theta"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_M"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_nuE"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_repN"])
    f.write("\n")
    f.write("\n")
    f.write(list_inputs["_D"])
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
    f.write(results_table.replace("\\begin{table}", "\\begin{table} \\n \\centering"))
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
    list_inputs["_essenceX"] = "Es(\\mathbf{X}) = " + array_to_matrix(model.essence_design_matrix)
    list_inputs["_B"] = "\\mathbf{B} = " + array_to_matrix(model.hypothesis_beta)
    list_inputs["_C"] = "\\mathbf{C} = " + array_to_matrix(model.c_matrix)
    list_inputs["_U"] = "\\mathbf{U} = " + array_to_matrix(model.u_matrix)
    list_inputs["_M"] = "\\mathbf{M} = " + array_to_matrix(model.m)
    list_inputs["_Sigma_Star"] = model.sigma_star
    list_inputs["_ThetaO"] = "\\mathbf{\\Theta}_{0} = " + array_to_matrix(model.theta_zero)
    list_inputs["_Theta"] = "\\mathbf{\\Theta} = " + array_to_matrix(model.theta)
    list_inputs["_nuE"] = "$\\nu_e = " + str(model.nu_e)
    list_inputs["_repN"] = "Number of repeated rows in design matrix = " + str(model.repeated_rows_in_design_matrix)
    list_inputs["_D"] = "Es(\mathbf{\Delta}) = " + array_to_matrix(model.delta)

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

    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name, True)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['lower confidence interval v2', 'upper confidence interval v2', 'Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_lower_v2_v3"].max(),
                                                  _df_output["deviation_upper_v2_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


def get_print_output_with_concat(_df_v2results, _df_vtest, output_name, confidence_limits=False):
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
    _df_output["Power_v2"] = _df_output["Power_v2"].apply(lambda x: '%.7f' % x)
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
    if confidence_limits:
        _df_output['deviation_lower_v2_v3'] = abs(_df_output[_df_output.columns[1]] - _df_output.lower_v2).apply(lambda x: '%.7f' % x)
        _df_output['deviation_upper_v2_v3'] = abs(_df_output[_df_output.columns[2]] - _df_output.upper_v2).apply(lambda x: '%.7f' % x)
        _df_output[_df_output.columns[1]] = _df_output[_df_output.columns[1]].apply(lambda x: '%.7f' % x)
        _df_output[_df_output.columns[2]] = _df_output[_df_output.columns[2]].apply(lambda x: '%.7f' % x)

        _df_output["GLIMMPSE V3 lower (deviation)"] = _df_output[_df_output.columns[1]].astype('str') + \
                                                      " (" + _df_output["deviation_lower_v2_v3"].astype('str') + ")"
        _df_output["GLIMMPSE V3 upper (deviation)"] = _df_output[_df_output.columns[2]].astype('str') + \
                                                      " (" + _df_output["deviation_upper_v2_v3"].astype('str') + ")"

    _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'SAS Power (deviation)', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'Test',
         'Sigma Scale', 'Beta Scale', 'Total N',
         'Alpha', 'Time_v3']]
    if confidence_limits:
        _df_print = _df_output[
        ['GLIMMPSE V3 Power', 'GLIMMPSE V3 lower (deviation)', 'GLIMMPSE V3 upper (deviation)', 'SAS Power (deviation)', 'Sim Power (deviation)', 'GLIMMPSE V2 Power (deviation)', 'lower_v2', 'upper_v2', 'Test',
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
                               'Max. Deviation': [_df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
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
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_with_confidence_intervals(file_path + model) for model in V3_JSON],
                          ignore_index=True).sort_values(['Test', 'Total N', 'Power'], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results, skipfooter=9, engine='python',
                                na_values=('NaN', 'n/a', ' n/a'))
    _df_v2results = _df_v2results[_df_v2results['Test'] != 0]
    _df_v2results = _df_v2results[_df_v2results['Total N'].notna()]
    _df_v2results = _df_v2results.sort_values(['Test', 'Total N', 'Power_v2'], ignore_index=True)
    _df_output = pd.concat([_df_vtest, _df_v2results], axis=1)

    _df_output['deviation_v2_v3'] = abs(_df_output.Power_v3 - _df_output.Power_v2).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sas_v3'] = abs(_df_output.Power_v3 - _df_output.SAS_Power).apply(lambda x: '%.7f' % x)
    _df_output['deviation_sim_v3'] = abs(_df_output.Power_v3 - _df_output.Sim_Power).apply(lambda x: '%.7f' % x)


    _df_output.to_excel(output_name + '.xlsx')

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

def tex_table_by_delta(file_path, output_name, V3_JSON: [], V2_results):
    list_inputs = get_inputs(V3_JSON, file_path)
    _df_vtest = pd.concat([json_power_by_delta(file_path + model) for model in V3_JSON], ignore_index=True).add_suffix("_v3")
    _df_v2results = pd.read_csv(file_path + V2_results,skipfooter=11, engine='python', na_values=('nan', 'NaN', 'n/a', ' n/a'))
    _df_output=pd.concat([_df_vtest, _df_v2results], axis=1)
    _df_output = _df_output[_df_output['SAS_Power'].notna()]
    _df_output.to_excel(output_name + '.xlsx')
    _df_v2summary = get_summary_results(V2_results, _df_v2results, file_path)
    _df_output, _df_print = get_print_output_with_concat(_df_v2results, _df_vtest, output_name, True)

    timings = pd.DataFrame({'Timing for': ['GLIMMPSE V3', 'GLIMMPSE V2', 'Simulation'],
                            'Total': [_df_output["Time"].sum(), _df_v2summary['Name']['Total Calculation Time'],
                                      _df_v2summary['Name']['Total Simulation Time']],
                            'Mean': [_df_output["Time"].mean(), _df_v2summary['Name']['Mean Calculation Time'],
                                     _df_v2summary['Name']['Mean Simulation Time']]})
    deviations = pd.DataFrame({'GLIMMPSE V3 deviation from': ['lower confidence interval v2', 'upper confidence interval v2', 'Simulation', 'SAS', 'GLIMMPSE V2'],
                               'Max. Deviation': [_df_output["deviation_lower_v2_v3"].max(),
                                                  _df_output["deviation_upper_v2_v3"].max(),
                                                  _df_output["deviation_sas_v3"].max(),
                                                  _df_output["deviation_sim_v3"].max(),
                                                  _df_output["deviation_v2_v3"].max()]})

    return timings.to_latex(index=False), deviations.to_latex(index=False), _df_print.to_latex(index=False,
                                                                                               longtable=True), list_inputs


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
test6_timings, test6_deviations, test6_results, test6_list_inputs = tex_table_by_delta(file_path, TEST_6_FILENAME, ['Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.json'], 'Example_6_Power_and_confidence_limits_for_the_univariate_approach_to_repeated_measures_in_a_multivariate_model.csv')
test7_timings, test7_deviations, test7_results, test7_list_inputs = tex_table_test7(file_path, TEST_7_FILENAME, ['Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.json'], 'Example_7_Power_for_a_time_by_treatment_interaction_using_orthogonal_polynomial_contrast_for_time.csv')

gaussian_test1_timings, gaussian_test1_deviations, gaussian_test1_results, gaussian_test1_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_1_FILENAME, ['GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_1_Median_power_for_the_Hotelling-Lawley_Trace_using_the_Satterthwaite_approximation_part_3.json'], 'Example_1_Median_power_for_the_HotellingLawley_Trace_using_the_Satterthwaite_approximation.csv')
gaussian_test4_timings, gaussian_test4_deviations, gaussian_test4_results, gaussian_test4_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_4_FILENAME, ['GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_2.json','GLMM_F_g_Example_4_Unconditional_power_for_the_Hotelling-Lawley_Trace_using_Davies_Algorithm_part_3.json'], 'Example_4_Unconditional_power_for_the_HotellingLawley_Trace_using_Davies_algorithm.csv')
gaussian_test5_timings, gaussian_test5_deviations, gaussian_test5_results, gaussian_test5_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_5_FILENAME,['GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_1.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_2.json', 'GLMM_F_g_Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser_Greenhouse_and_Huynh-Feldt_tests_using_the_Satterthwaite_approximation_part_3.json'], 'Example_5_Median_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_GeisserGreenhouse_and_HuynhFeldt_tests_using_the_Satterthwaite_approximation.csv')
gaussian_test8_timings, gaussian_test8_deviations, gaussian_test8_results, gaussian_test8_list_inputs = tex_table_gaussian(file_path, GAUSSIAN_TEST_8_FILENAME, ['GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_1.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_2.json', 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_Algorithm_part_3.json'], 'GLMM_F_g_Example_8_Unconditional_power_for_the_uncorrected_univariate_approach_to_repeated_measures_Box_Geisser-Greenhouse_and_Huynh-Feldt_tests_using_Davies_algorithm.csv')

TEST_1_TITLE = """GLMM(F) Example 1. Power for a two sample t-test for several error variance values and mean differences"""
TEST_1_STUDY_DESIGN_DESCRIPTION ="""The study design for Example 1 is a balanced, two-group design.
We calculate power for a two-sample t-test comparing the mean responses between the two groups.
The example is based on the results in  Muller, K. E., \\& Benignus, V. A. (1992). \\emph{Neurotoxicology and teratology}, \\emph{14}(3), 211-219."""

TEST_2_TITLE = """GLMM(F) Example 2. Power results for a Paired T-test"""
TEST_2_STUDY_DESIGN_DESCRIPTION = """The study design in Example 2 is a one sample design with a pre- and post-measurement for each participant.
We calculate power for a paired t-test comparing the mean responses at the pre- and post-measurements. We
express the paired t-test as a general linear hypothesis in a multivariate linear model."""

TEST_3_TITLE="""GLMM(F) Example 3. Power for a two sample t-test for various sample sizes and mean differences"""
TEST_3_STUDY_DESIGN_DESCRIPTION="""The study design for Example 3 is a balanced, two sample design with a single response variable. We calculate
power for a two-sample t-test comparing the mean responses between the two independent groups. The example
demonstrates changes in power with different sample sizes and mean differences."""

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

TEST_6_TITLE="""GLMM(F) Example 6. Power and condence limits for the univariate approach to repeated measures in a multivariate model"""
TEST_6_STUDY_DESIGN_DESCRIPTION="""The study design for Example 6 is a factorial design with two between participant factors and one within participant factor. Participants were categorized by gender and classied into ve age groups. For each participant,
cerebral vessel tortuosity was measured in four regions of the brain. We calculate power for a test of the gender
by region interaction. Confidence limits are computed for the power values.
The matrix inputs below show the starting point for the B matrix. The third column of the matrix (i.e. vessel
tortuosity in the third region the brain) is modied throughout the validation experiment to progressively increase
the eect of gender. Mean values for males are increased by 0.0008 at each iteration, while corresponding values
for females are decremented by 0.0008. The process is restarted for each statistical test. For example, the first
power calculated for a given test would use the following B matrix 

\\mathbf{B} = \\begin{bmatrix}
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
\\end{bmatrix} + \\begin{bmatrix}
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & 0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
0 & 0 & -0.0008 & 0 \\\\
\\end{bmatrix}

= \\begin{bmatrix}
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.5008 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
2.9 & 3.2 & 3.4992 & 3.2 \\\\
\\end{bmatrix}

And the last power calculated for a given test would use the following B matrix 

\\mathbf{B} = \\begin{bmatrix}
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
2.9 & 3.2 & 3.5 & 3.2 \\\\
\\end{bmatrix} + \\begin{bmatrix}
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & 0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
0 & 0 & -0.2 & 0 \\\\
\\end{bmatrix}

= \\begin{bmatrix}
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.7 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
2.9 & 3.2 & 3.3 & 3.2 \\\\
\\end{bmatrix}


This is based on an example presented in
Gurka, M. J., Coey, C. S., & Muller, K. E. (2007). Internal pilots for a class of linear mixed models with
Gaussian and compound symmetric data. Statistics in Medicine, 26(22), 4083-4099.
"""

TEST_7_TITLE="""Example 7. Power for a time by treatment interaction
using orthogonal polynomial contrast for time"""
TEST_7_STUDY_DESIGN_DESCRIPTION="""The study design for Example 7 is a balanced two sample design with  ve repeated measures over time. We
calculate power for a test of the time trend by treatment interaction. The example demonstrates the use of an
orthogonal polynomial contrast for the e ect of time."""

GAUSSIAN_TEST_1_TITLE="""GLMM(F, g) Example 1. Median power for the Hotelling-Lawley
Trace, using the Satterthwaite approximation"""
GAUSSIAN_TEST_1_STUDY_DESIGN_DESCRIPTION="""The study design in Example 1 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the median power for a test of no di erence between groups at each time point, using the Hotelling-
Lawley Trace test. A Satterthwaite approximation is used to obtain the approximate distribution of the test
statistic under the alternative hypothesis. Median power is calculated for the following combinations of mean
di erences and per group sample sizes.


1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835


The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_4_TITLE="""GLMM(F, g) Example 4. Unconditional power for the Hotelling-
Lawley Trace, using Davies algorithm"""
GAUSSIAN_TEST_4_STUDY_DESIGN_DESCRIPTION="""The study design in Example 4 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the unconditional power for a test of no di erence between groups at each time point, using the
Hotelling-Lawley Trace test. The exact distribution of the test statistic under the alternative hypothesis is
obtained using Davies' algorithm described in
Davies, R. B. (1980). Algorithm AS 155: The Distribution of a Linear Combination of Chi-Square Random
Variables. Applied Statistics, 29(3), 323-333.
Unconditional power is calculated for the following combinations of mean di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_5_TITLE="""GLMM(F, g) Example 5. Median power for the uncorrected univari-
ate approach to repeated measures, Box, Geisser-Greenhouse, and
Huynh-Feldt tests, using the Satterthwaite approximation"""
GAUSSIAN_TEST_5_STUDY_DESIGN_DESCRIPTION="""The study design in Example 5 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the median power for a test of no di erence between groups at each time point. We calculate
median power for the uncorrected univariate approach to repeated measures, Box, Geisser-Greenhouse, and
Huynh-Feldt tests.A Satterthwaite approximation is used to obtain the approximate distribution of the test
statistic under the alternative hypothesis. Median power is calculated for the following combinations of mean
di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

GAUSSIAN_TEST_8_TITLE="""GLMM(F, g) Example 8. Unconditional power for the uncorrected
univariate approach to repeated measures, Box, Geisser-Greenhouse,
and Huynh-Feldt tests, using Davies algorithm"""
GAUSSIAN_TEST_8_STUDY_DESIGN_DESCRIPTION="""The study design in Example 8 is a three sample design with a baseline covariate and four repeated measurements.
We calculate the unconditional power for a test of no di erence between groups at each time point. We calculate
unconditional power for the uncorrected univariate approach to repeated measures, Box, Geisser-Greenhouse,
and Huynh-Feldt tests.The exact distribution of the test statistic under the alternative hypothesis is obtained
using Davies' algorithm described in
Davies, R. B. (1980). Algorithm AS 155: The Distribution of a Linear Combination of Chi-Square Random
Variables. Applied Statistics, 29(3), 323-333.
Unconditional power is calculated for the following combinations of mean di erences and per group sample sizes.

1. Per group sample size of 5, with beta scale values 0.4997025, 0.8075886, and 1.097641
2. Per group sample size of 25, with beta scale values 0.1651525, 0.2623301, and 0.3508015
3. Per group sample size of 50, with beta scale values 0.1141548, 0.1812892, and 0.2423835

The example is based on Table II from
Glueck, D. H., & Muller, K. E. (2003). Adjusting power for a baseline covariate in linear models. Statistics in
Medicine, 22(16), 2535-2551."""

INTRODUCTION = """The following report contains validation results for the JavaStatistics library, a component of the GLIMMPSE
software system. For more information about GLIMMPSE and related publications, please visit

https://samplesizeshop.org.

The automated validation tests shown below compare power values produced by the GLIMMPSE V3 to
published results and also to simulation. Sources for published values include POWERLIB (Johnson et al. 2007)
and a SAS IML implementation of the methods described by Glueck and Muller (2003).
Validation results are listed in Section 3 of the report. Timing results show the calculation and simulation times
for the overall experiment and the mean times per power calculation. Summary statistics show the maximum
absolute deviation between the power value calculated by GLIMMPSE V3, the JavaStatistics library and the results obtained from
SAS or via simulation. The table in Section 3.3 shows the deviation values for each individual power comparison."""


all_names = [TEST_1_FILENAME, TEST_2_FILENAME, TEST_3_FILENAME, TEST_4_FILENAME, TEST_5_FILENAME, TEST_6_FILENAME, TEST_7_FILENAME]
all_titles = [TEST_1_TITLE, TEST_2_TITLE, TEST_3_TITLE, TEST_4_TITLE, TEST_5_TITLE, TEST_6_TITLE, TEST_7_TITLE]
all_descriptions = [TEST_1_STUDY_DESIGN_DESCRIPTION, TEST_2_STUDY_DESIGN_DESCRIPTION, TEST_3_STUDY_DESIGN_DESCRIPTION, TEST_4_STUDY_DESIGN_DESCRIPTION, TEST_5_STUDY_DESIGN_DESCRIPTION, TEST_6_STUDY_DESIGN_DESCRIPTION, TEST_7_STUDY_DESIGN_DESCRIPTION]
all_list_inputs = [test1_list_inputs, test2_list_inputs, test3_list_inputs, test4_list_inputs, test5_list_inputs, test6_list_inputs, test7_list_inputs]
all_timings = [test1_timings, test2_timings, test3_timings, test4_timings, test5_timings,  test6_timings, test7_timings]
all_deviations = [test1_deviations, test2_deviations, test3_deviations, test4_deviations, test5_deviations, test6_deviations, test7_deviations]
all_results = [test1_results, test2_results, test3_results, test4_results, test5_results, test6_results, test7_results]


all_gaussian_names = [GAUSSIAN_TEST_1_FILENAME, GAUSSIAN_TEST_4_FILENAME, GAUSSIAN_TEST_5_FILENAME, GAUSSIAN_TEST_8_FILENAME]
all_gaussian_titles = [GAUSSIAN_TEST_1_TITLE,GAUSSIAN_TEST_4_TITLE,GAUSSIAN_TEST_5_TITLE,GAUSSIAN_TEST_8_TITLE,]
all_gaussian_descriptions = [GAUSSIAN_TEST_1_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_4_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_5_STUDY_DESIGN_DESCRIPTION,GAUSSIAN_TEST_8_STUDY_DESIGN_DESCRIPTION,]
all_gaussian_list_inputs = [gaussian_test1_list_inputs,gaussian_test4_list_inputs,gaussian_test5_list_inputs,gaussian_test8_list_inputs,]
all_gaussian_timings = [gaussian_test1_timings,gaussian_test4_timings,gaussian_test5_timings,gaussian_test8_timings,]
all_gaussian_deviations = [gaussian_test1_deviations,gaussian_test4_deviations,gaussian_test5_deviations,gaussian_test8_deviations,]
all_gaussian_results = [gaussian_test1_results,gaussian_test4_results,gaussian_test5_results,gaussian_test8_results,]


for n in all_names:
    write_tex_file(False,
                   n,
                   all_titles[all_names.index(n)],
                   INTRODUCTION,
                   all_descriptions[all_names.index(n)],
                   all_list_inputs[all_names.index(n)],
                   all_timings[all_names.index(n)],
                   all_deviations[all_names.index(n)],
                   all_results[all_names.index(n)])
    write_pdf(n)

for n in all_gaussian_names:
     write_tex_file(True,
                    n,
                    all_gaussian_titles[all_gaussian_names.index(n)],
                    INTRODUCTION,
                    all_gaussian_descriptions[all_gaussian_names.index(n)],
                    all_gaussian_list_inputs[all_gaussian_names.index(n)],
                    all_gaussian_timings[all_gaussian_names.index(n)],
                    all_gaussian_deviations[all_gaussian_names.index(n)],
                    all_gaussian_results[all_gaussian_names.index(n)])
     write_pdf(n)
