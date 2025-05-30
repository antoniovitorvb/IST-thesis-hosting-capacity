import os, math
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control as ppc
import create_basic_network as cbn
from copy import deepcopy

from pandapower.file_io import from_json, to_json

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')
json_dir = os.path.join(os.path.dirname(__file__), 'json_networks')

ds = cbn.create_data_source(data_dir, profile_dir=os.path.join(data_dir, 'Load Profiles'))
print(f"Created {ds}")

net = from_json(os.path.join(json_dir, "no_load_network.json"))

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
            profile_name=f"CTRL_{row['Name']}", data_source=ds
        )
        # print(f"Created {net.controller.object.iloc[load+1].profile_name}")

