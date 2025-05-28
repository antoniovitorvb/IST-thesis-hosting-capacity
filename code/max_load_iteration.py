import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from pandapower.file_io import from_json, to_json
from create_basic_network import debug_result
from itertools import combinations as comb

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')

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
to_json(net, os.path.join('json_networks', f"{max_batch}_loads_network.json"))