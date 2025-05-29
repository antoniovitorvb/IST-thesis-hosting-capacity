import os, math
import pandas as pd
import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt

from pandapower.file_io import from_json, to_json
from create_basic_network import debug_result

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')
json_dir = os.path.join(os.path.dirname(__file__), 'json_networks')

net = from_json(os.path.join(json_dir, "no_load_network.json"))

if debug_result(net, init='auto'): print("Debugging successful")

loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)

farthest_load = loads_df.loc[loads_df['Bus'].astype(int).idxmax()]

pa = qa = pb = qb = pc = qc = 0
if farthest_load['phases']=='A':
    pa = farthest_load['kW']
    qa = farthest_load['kW'] * np.tan(np.arccos(farthest_load['PF']))
elif farthest_load['phases']=='B':
    pb = farthest_load['kW']
    qb = farthest_load['kW'] * np.tan(np.arccos(farthest_load['PF']))
else: # farthest_load['phases']=='C'
    pc = farthest_load['kW']
    qc = farthest_load['kW'] * np.tan(np.arccos(farthest_load['PF']))

pp.create_asymmetric_load(
    net, bus=math.floor(farthest_load['Bus']),
    p_a_mw=pa / 1000, q_a_mvar=qa / 1000,
    p_b_mw=pb / 1000, q_b_mvar=qb / 1000,
    p_c_mw=pc / 1000, q_c_mvar=qc / 1000,
    name=farthest_load['Name']
)
print(f"\nCreated one {farthest_load['kW']} kW load at bus {math.floor(farthest_load['Bus'])} at phase {farthest_load['phases']}:\n")

print("\nSaving network with farthest load to JSON...\n")
to_json(net, os.path.join(json_dir, "farthest_load_network.json"))

inc = 5e-5
success = True
vm_pu = pd.DataFrame(columns=['p_a_mw', 'vm_a_pu', 'vm_b_pu', 'vm_c_pu'])
print("\nStarting load increment...\n")
while success:
    # print(net.asymmetric_load.p_a_mw)
    if debug_result(net, init='flat'):
        new_row = pd.concat([
            net.asymmetric_load.p_a_mw,
            net.res_bus_3ph.loc[net.asymmetric_load.bus, ['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].reset_index(drop=True)
        ], axis=1)
        vm_pu = pd.concat([vm_pu, new_row], ignore_index=True)

        net.asymmetric_load.p_a_mw += inc
    else:
        print(f"\nMax P = {net.asymmetric_load.p_a_mw[0]*1000:.2f} kW")
        success = False

plt.figure(figsize=(10, 6))

plt.fill_between(vm_pu['p_a_mw']*1000,
                 y1=vm_pu['vm_a_pu'].max(), y2=0.95,
                #  where=vm_pu['vm_a_pu'] < 0.95, 
                 interpolate=True, color='green', alpha=0.1)
plt.fill_between(vm_pu['p_a_mw']*1000,
                 y1=0.95, y2=0.9,
                #  where=vm_pu['vm_a_pu'] < 0.95, 
                 interpolate=True, color='red', alpha=0.1)
plt.fill_between(vm_pu['p_a_mw']*1000,
                 y1=0.9, y2=vm_pu['vm_a_pu'].min(),
                #  where=vm_pu['vm_a_pu'] < 0.95, 
                 interpolate=True, color='darkred', alpha=0.2)
plt.hlines(y=0.95, xmin=vm_pu['p_a_mw'].min()*1000,
           xmax=vm_pu['p_a_mw'].max()*1000, color='red', linestyle='--', label='Limit 0.95 pu')
plt.hlines(y=0.9, xmin=vm_pu['p_a_mw'].min()*1000,
           xmax=vm_pu['p_a_mw'].max()*1000, color='darkred', linestyle='--', label='Limit 0.90 pu')

plt.plot(vm_pu['p_a_mw']*1000, vm_pu['vm_a_pu'], label='Va (pu)', marker='o')
# plt.title('Voltage Deviation with Increasing Load from Farthest Bus')
plt.xlabel('Load (kW)')
plt.ylabel('Voltage Magnitude (pu)')
plt.legend()

plt.savefig(os.path.join('images', 'farthest_load_X_voltage.png'))