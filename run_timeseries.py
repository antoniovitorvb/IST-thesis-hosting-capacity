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

profile_df, ds = cbn.create_data_source(data_dir, profile_dir=os.path.join(data_dir, 'Load Profiles'), ds_index=True)
print(f"Created {ds} {ds.df.shape}")

net = from_json(os.path.join(json_dir, "full_load_network.json"))

loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)
loadShape_df = pd.read_csv(os.path.join(data_dir, 'LoadShapes.csv'), skiprows=1, sep=';')
loads_df.head(), loadShape_df.head()

merged_load = loads_df.merge(loadShape_df, left_on="Yearly", right_on="Name", how="left")


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