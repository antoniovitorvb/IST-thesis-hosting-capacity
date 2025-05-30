import os, math
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control as ppc
import create_basic_network as cbn
from copy import deepcopy

from pandapower.file_io import from_json, to_json
from pandapower.timeseries import run_timeseries
from pandapower.timeseries import OutputWriter

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')
json_dir = os.path.join(os.path.dirname(__file__), 'json_networks')

ds = cbn.create_data_source(data_dir, profile_dir=os.path.join(data_dir, 'Load Profiles'), ds_index=True)[1]
print(f"Created {ds} {ds.df.shape}")

net = from_json(os.path.join(json_dir, "no_load_network.json"))
try:
    pp.runpp_3ph(net, init='auto', max_iteration=100, tolerance_mva=1e-5, calc_voltage_angles=True, v_debug=True)
    print("\nDebugging successful!")
except Exception as e:
    print(e)

loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)
loadShape_df = pd.read_csv(os.path.join(data_dir, 'LoadShapes.csv'), skiprows=1, sep=';')
loads_df.head(), loadShape_df.head()

merged_load = loads_df.merge(loadShape_df, left_on="Yearly", right_on="Name", how="left")
# print(merged_load.head(10))

load_map = {}

for _, row in loads_df.iterrows():
    bus_id = math.floor(row['Bus'])
    pa = qa = pb = qb = pc = qc = 0

    if (bus_id in net.bus.name) and (row['phases'] in {'A', 'B', 'C'}):
        if row['phases'] == 'A':
            pa = row['kW']
            qa = row['kW'] * np.tan(np.arccos(row['PF']))
        elif row['phases'] == 'B':
            pb = row['kW']
            qb = row['kW'] * np.tan(np.arccos(row['PF']))
        else:  # 'C'
            pc = row['kW']
            qc = row['kW'] * np.tan(np.arccos(row['PF']))

        load = pp.create_asymmetric_load(
            net, bus=net.bus.index[net.bus.name == bus_id][0],
            p_a_mw=pa / 1000, q_a_mvar=qa / 1000,
            p_b_mw=pb / 1000, q_b_mvar=qb / 1000,
            p_c_mw=pc / 1000, q_c_mvar=qc / 1000,
            name=row['Name']
        )
        load_map[load] = row['Yearly']

        # Attach ConstControl
        phase = row['phases'].lower()
        ppc.ConstControl(
            net, element_index=load,
            variable=f"p_{phase}_mw",
            element="asymmetric_load",
            profile_name=row['Name'], data_source=ds
        )
        # print(f"Created {net.controller.object.iloc[load+1].profile_name}")

output_dir = os.path.join(os.path.dirname(__file__), "timeseries_results_py")
if not os.path.exists(output_dir): os.mkdir(output_dir)

# Create and attach output writer to net48
ow = OutputWriter(net, time_steps=ds.df.index, output_path=output_dir)
# Log desired variables
ow.log_variable('res_trafo_3ph', 'loading_percent', index=net.trafo.index[0])

ow.log_variable('res_bus_3ph', 'vm_a_pu', index=net.bus.index)
ow.log_variable('res_bus_3ph', 'vm_b_pu', index=net.bus.index)
ow.log_variable('res_bus_3ph', 'vm_c_pu', index=net.bus.index)

ow.log_variable('res_line_3ph', 'loading_a_percent', index=net.line.index)
ow.log_variable('res_line_3ph', 'loading_b_percent', index=net.line.index)
ow.log_variable('res_line_3ph', 'loading_c_percent', index=net.line.index)

run_timeseries(net, time_steps=ds.df.index, run=pp.runpp_3ph, run_control=True, continue_on_divergence=False, verbose=True)