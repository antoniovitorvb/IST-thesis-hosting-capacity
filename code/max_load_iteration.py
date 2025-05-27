import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from pandapower.file_io import from_json, to_json
from itertools import combinations as comb

data_dir = os.path.join(os.path.dirname(__file__),'Modified_116_LV_CSV')

def export_results(file_path, net, init='auto', max_iteration=100, tolerance_mva=1e-8):
    """
    Export all result DataFrames (attributes of net starting with 'res_') to an Excel file.
    
    Parameters:
    - file_path: str, path to the Excel file to save the results.
    - net: object containing result DataFrames as attributes starting with 'res_'.
    """

    try:
        pp.runpp_3ph(
            net, init=init,
            max_iteration=max_iteration,
            tolerance_mva=tolerance_mva,
            calc_voltage_angles=True,
            v_debug=True
        )
    except Exception as e:
        return e  # or raise e if you want the exception to propagate

    with pd.ExcelWriter(file_path) as writer:
        for attr in dir(net):
            if attr.startswith("res_"):
                df = getattr(net, attr)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sheet_name = attr[:31]  # Excel sheet names must be <= 31 characters
                    df.to_excel(writer, sheet_name=sheet_name)
    return True

def debug_result(net, init='auto', max_iteration=100, tolerance_mva=1e-8):
    try:
        pp.runpp_3ph(
            net, init=init,
            max_iteration=max_iteration,
            tolerance_mva=tolerance_mva,
            calc_voltage_angles=True,
            v_debug=True
        )
    except Exception as e:
        return False
    return True

net = from_json(os.path.join(data_dir, "no_load_network.json"))
sample_net = from_json(os.path.join(data_dir, "no_load_network.json"))
loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)

if debug_result(net, init='auto', max_iteration=100, tolerance_mva=1e-8): print("Debugging successful")

N = 1000
batch = range(50, len(loads_df)+1)
max_batch = 0

for b in batch:
    N = min(len(list(comb(range(len(loads_df)), b))), 1000)
    print(f"Processing batch {b}")

    for i in range(N):
        sample_loads = loads_df.sample(b)

        sample_net.asymmetric_load.drop(sample_net.asymmetric_load.index, inplace=True)
        for _, row in sample_loads.iterrows():
            # print(_, '->', row)
            bus_id = math.floor(row['Bus'])
            pa = qa = pb = qb = pc = qc = 0
            if row['phases'] in {'A', 'B', 'C'}:
                if row['phases']=='A':
                    pa = row['kW']
                    qa = row['kW'] * np.tan(np.arccos(row['PF']))
                elif row['phases']=='B':
                    pb = row['kW']
                    qb = row['kW'] * np.tan(np.arccos(row['PF']))
                else: # row['phases']=='C'
                    pc = row['kW']
                    qc = row['kW'] * np.tan(np.arccos(row['PF']))
                load = pp.create_asymmetric_load(
                    sample_net, bus=sample_net.bus.index[sample_net.bus.name==bus_id][0],
                    p_a_mw=pa, q_a_mvar=qa,
                    p_b_mw=pb, q_b_mvar=qb,
                    p_c_mw=pc, q_c_mvar=qc,
                    name=row['Name']
                )

        if debug_result(sample_net, init='flat'):
            print(f"\n{b} Batch power Flow Sucessful!\n")
            max_batch = b
            net.asymmetric_load.drop(net.asymmetric_load.index, inplace=True)
            for _, row in sample_loads.iterrows():
                # print(_, '->', row)
                bus_id = math.floor(row['Bus'])
                pa = qa = pb = qb = pc = qc = 0
                if row['phases'] in {'A', 'B', 'C'}:
                    if row['phases']=='A':
                        pa = row['kW']
                        qa = row['kW'] * np.tan(np.arccos(row['PF']))
                    elif row['phases']=='B':
                        pb = row['kW']
                        qb = row['kW'] * np.tan(np.arccos(row['PF']))
                    else: # row['phases']=='C'
                        pc = row['kW']
                        qc = row['kW'] * np.tan(np.arccos(row['PF']))
                    load = pp.create_asymmetric_load(
                        net, bus=net.bus.index[net.bus.name==bus_id][0],
                        p_a_mw=pa, q_a_mvar=qa,
                        p_b_mw=pb, q_b_mvar=qb,
                        p_c_mw=pc, q_c_mvar=qc,
                        name=row['Name']
                    )
            break

print(net)
to_json(net, os.path.join(data_dir, f"{max_batch}_loads_network.json"))