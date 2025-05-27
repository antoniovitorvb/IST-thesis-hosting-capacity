import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from pandapower.file_io import from_json, to_json

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

if debug_result(net, init='auto', max_iteration=100, tolerance_mva=1e-8): print("Debugging successful")

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
print(net)

# to_json(net, os.path.join(data_dir, "farthest_load_network.json"))

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

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

plt.plot(vm_pu['p_a_mw']*1000, vm_pu['vm_a_pu'], label='Phase A', marker='o')
# plt.title('Load (kW) X Voltage Magnitude (pu)')
plt.xlabel('Load (kW)')
plt.ylabel('Voltage Magnitude (pu)')

print('Saving plot...')
plt.savefig(os.path.join('images', 'farthest_load_X_voltage.png'))