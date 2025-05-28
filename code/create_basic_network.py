import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from pandapower.file_io import from_json, to_json
from max_i_pred import max_i_pred

data_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')

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

def hc_violation(net, mod='det'):
    if mod == 'det': vm_max, vm_min = [1.05, 0.95]
    elif mod == 'sto': vm_max, vm_min = [1.10, 0.9]
    else: raise ValueError("Invalid mode. Use 'det' or 'stoch'.")
    # pp.runpp(net)
    
    return any([
        net.res_trafo_3ph.loading_percent.max() > 100, # 110
        net.res_line_3ph.loading_a_percent.max() > 100, # 110
        net.res_line_3ph.loading_b_percent.max() > 100,
        net.res_line_3ph.loading_c_percent.max() > 100,
        net.res_bus_3ph.vm_a_pu.max() >= vm_max,
        net.res_bus_3ph.vm_b_pu.max() >= vm_max,
        net.res_bus_3ph.vm_c_pu.max() >= vm_max,
        net.res_bus_3ph.vm_a_pu.min() <= vm_min,
        net.res_bus_3ph.vm_b_pu.min() <= vm_min,
        net.res_bus_3ph.vm_c_pu.min() <= vm_min
    ])

# Source data
source_df = pd.read_csv(os.path.join(data_dir,'Source.csv'), skiprows=1, sep='=')
source_dict = {i:float(row.iloc[0].split()[0]) for i, row in source_df.iterrows()}
source_dict['ISC1'] = source_dict['ISC1'] / 1000
source_dict['ISC3'] = source_dict['ISC3'] / 1000

# Transformer data
trafo_df = pd.read_csv(os.path.join(data_dir,'Transformer.csv'), skiprows=1, sep=';')
trafo_dict = trafo_df.iloc[0].to_dict()

s_sc_mva = np.sqrt(3) * source_dict['Voltage'] * source_dict['ISC3'] #MVA

net = pp.create_empty_network()

bus_map = {}

hv_bus = pp.create_bus(net, name=trafo_dict[' bus1'], vn_kv=source_dict['Voltage'], type="b")
lv_bus = pp.create_bus(net, name=trafo_dict[' bus2'], vn_kv=trafo_dict[' kV_sec'], type="b")

bus_map[trafo_dict[' bus1']] = hv_bus
bus_map[trafo_dict[' bus2']] = lv_bus

pp.create_ext_grid(
    net, bus=hv_bus, vm_pu=source_dict['pu'],
    s_sc_max_mva=s_sc_mva,
    rx_max=0.1, rx_min=None,
    # max_p_mw=None, min_p_mw=None,
    # max_q_mvar=None, min_q_mvar=None, index=None,
    r0x0_max=0.1, x0x_max=1.0
)
print(f"\nCreated External Grid!\n")

pp.create_transformer_from_parameters(
    net=net, hv_bus=hv_bus, lv_bus=lv_bus,
    sn_mva=trafo_dict[' MVA'],
    vn_hv_kv=trafo_dict[' kV_pri'],
    vn_lv_kv=trafo_dict[' kV_sec'],
    vk_percent=trafo_dict[' %XHL'],
    vkr_percent=trafo_dict['% resistance'],
    vk0_percent=trafo_dict[' %XHL'],
    vkr0_percent=trafo_dict['% resistance'],
    mag0_percent=100, mag0_rx=0,
    si0_hv_partial=0.9,
    pfe_kw=0.0, i0_percent=0.0, shift_degree=30, 
    name=trafo_dict['Name'], vector_group='Dyn',
)
print(f"\nCreated Transformer!\n")

pp.create_shunt(
    net, bus=lv_bus,
    q_mvar=-0.01, p_mw=10e-3,
    vn_kv=trafo_dict[' kV_sec'],
    name="trafo_lv_shunt"
)
print(f"\nCreated Shunt!\n")

lines_df = pd.read_excel(os.path.join(data_dir, "Lines.xlsx"), skiprows=1)
lines_df['Length'] = lines_df['Length'] / 1000 # m to km
lines_df['Units'] = 'km'

all_bus_ids = np.unique(lines_df[['Bus1', 'Bus2']].values.ravel('K'))
for id in all_bus_ids:
    id = math.floor(id)
    if id not in bus_map.keys():
        bus = pp.create_bus(net, name=id, vn_kv=trafo_dict[' kV_sec'], type="b")
        bus_map[id] = bus
print(f"\nCreated {len(net.bus)} Buses!\n")

lineCodes_df = pd.read_csv(os.path.join(data_dir, "LineCodes.csv"), skiprows=1, sep=';')

lineCodes_df['max_i_ka'] = max_i_pred(lineCodes_df)

full_line_df = lines_df.merge(lineCodes_df, left_on="LineCode", right_on="Name", how="left")

for _, line in full_line_df.iterrows():
    # print(line)

    if (line['C0'] == 0): line['C0'] = 200 # nF/km
    if (line['C1'] == 0): line['C1'] = 200 # nF/km
    
    pp.create_line_from_parameters(
        net, from_bus = bus_map[math.floor(line['Bus1'])],
        to_bus = bus_map[math.floor(line['Bus2'])],
        length_km = line['Length'],
        r_ohm_per_km=line["R1"], r0_ohm_per_km=line["R1"] * 2,
        x_ohm_per_km=line["X1"], x0_ohm_per_km=line["X0"],
        c_nf_per_km=line["C1"], c0_nf_per_km=line["C0"],
        max_i_ka=line["max_i_ka"],
        name=line["Name_x"], type='cs',
    )
print(f"\nCreated {len(net.line)} Lines!\n")

os.mkdir('json_networks') if not os.path.exists('json_networks') else None
print(net)
to_json(net, os.path.join('json_networks', "no_load_network.json"))