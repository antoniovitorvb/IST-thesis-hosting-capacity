import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from pandapower.file_io import from_json, to_json
from create_basic_network import debug_result

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')
json_dir = os.path.join(os.path.dirname(__file__), 'json_networks')

net = from_json(os.path.join(json_dir, "no_load_network.json"))
sample_net = from_json(os.path.join(json_dir, "no_load_network.json"))
loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)

if debug_result(net, init='auto', max_iteration=100, tolerance_mva=1e-8): print("Debugging successful")

# batch = range(53, len(loads_df)+1)
batch = range(49, 53)
max_batch = 0

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

for b in batch:
    N = min(math.comb(len(loads_df), b), 500)
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
                    p_a_mw=pa / 1000, q_a_mvar=qa / 1000,
                    p_b_mw=pb / 1000, q_b_mvar=qb / 1000,
                    p_c_mw=pc / 1000, q_c_mvar=qc / 1000,
                    name=row['Name']
                )

        if debug_result(sample_net, init='auto'):
            print(f"{b} Batch power Flow Sucessful after {i} iterations!\n")
            max_batch = b
            break
        if i == N-1: print(f"{b} Batch power Flow failed!\n")

if max_batch > 0:
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
                p_a_mw=pa / 1000, q_a_mvar=qa / 1000,
                p_b_mw=pb / 1000, q_b_mvar=qb / 1000,
                p_c_mw=pc / 1000, q_c_mvar=qc / 1000,
                name=row['Name']
            )

print(net)
to_json(net, os.path.join('json_networks', f"{max_batch}_loads_network.json"))