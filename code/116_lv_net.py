import os, math
import pandas as pd
import pandapower as pp
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

data_dir = os.path.join(os.path.dirname(__file__),'Modified_116_LV_CSV')
# print(data_dir)

# Source data
source_df = pd.read_csv(os.path.join(data_dir,'Source.csv'), skiprows=1, sep='=')
source_dict = {i:float(row.iloc[0].split()[0]) for i, row in source_df.iterrows()}
print(source_dict)

source_dict['ISC1'] = source_dict['ISC1'] / 1000
source_dict['ISC3'] = source_dict['ISC3'] / 1000

# Transformer data
trafo_df = pd.read_csv(os.path.join(data_dir,'Transformer.csv'), skiprows=1, sep=';')
trafo_dict = trafo_df.iloc[0].to_dict()
print(trafo_dict)

s_sc_mva = np.sqrt(3) * source_dict['Voltage'] * source_dict['ISC3'] #MVA

# Create the pandapower network
net = pp.create_empty_network()

bus_map = {}

hv_bus = pp.create_bus(net, name=trafo_dict[' bus1'], vn_kv=source_dict['Voltage'], type="b")
lv_bus = pp.create_bus(net, name=trafo_dict[' bus2'], vn_kv=trafo_dict[' kV_sec'], type="b")

bus_map[trafo_dict[' bus1']] = hv_bus
bus_map[trafo_dict[' bus2']] = lv_bus

pp.create_ext_grid(net, bus=hv_bus, vm_pu=source_dict['pu'], s_sc_max_mva=s_sc_mva, rx_max=0.1)

pp.create_transformer_from_parameters(
    net=net, hv_bus=hv_bus, lv_bus=lv_bus,
    sn_mva=trafo_dict[' MVA'],
    vn_hv_kv=trafo_dict[' kV_pri'],
    vn_lv_kv=trafo_dict[' kV_sec'],
    vk_percent=trafo_dict[' %XHL'],
    vkr_percent=trafo_dict['% resistance'],
    pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, 
    name=trafo_dict['Name']
)

lines_df = pd.read_excel(os.path.join(data_dir, "Lines.xlsx"), skiprows=1)
lines_df['Length'] = lines_df['Length'] / 1000 # m to km
lines_df['Units'] = 'km'
lines_df.head()

# Create bus objects for all buses in the lines_df
print('\nCREATING BUSES...\n')
all_bus_ids = np.unique(lines_df[['Bus1', 'Bus2']].values.ravel('K'))
for id in all_bus_ids:
    if id not in bus_map.keys():
        bus = pp.create_bus(net, name=id, vn_kv=trafo_dict[' kV_sec'], type="b")
        bus_map[id] = bus

print('\nIMPORTING LINE CODES...\n')
StdLineCodes_df = pd.read_csv(os.path.join(data_dir, 'StandardLineCodes.csv'), sep=';')
lineCodes_df = pd.read_csv(os.path.join(data_dir, "LineCodes.csv"), skiprows=1, sep=';')

X = StdLineCodes_df[['r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km']]
Y = StdLineCodes_df['max_i_ka']

x_test = lineCodes_df[['Name', 'R1', 'X1', 'C1']].set_index('Name')
x_test.columns = X.columns

print('\nTRAINING KNN...\n')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_test_scaled = scaler.fit_transform(x_test)

knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_scaled, Y)

y_pred = knn.predict(x_test_scaled)
print('\tKNN Line Max I pred =', y_pred)

lineCodes_df['max_i_ka'] = y_pred

print('\nCREATING LINES...\n')
full_line_df = lines_df.merge(lineCodes_df, left_on="LineCode", right_on="Name", how="left")
full_line_df.head(10)

for _, line in full_line_df.iterrows():
    # print(line)
    pp.create_line_from_parameters(
        net, from_bus = line['Bus1'], to_bus = line['Bus2'],
        length_km = line['Length'],
        r_ohm_per_km=line["R1"],
        x_ohm_per_km=line["X1"],
        c_nf_per_km=line["C1"],
        max_i_ka=line["max_i_ka"],
        name=line["Name_x"]
    )

print('\nCREATING LOADS...\n')# Create Loads
loads_df = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)
print(loads_df.head(10))

# net.asymmetric_load.drop(net.asymmetric_load.index, inplace=True)
load_map = {}
for _, row in loads_df.iterrows():
    # print(_, '->', row)
    bus_id = math.floor(row['Bus'])
    pa = qa = pb = qb = pc = qc = 0
    if (bus_id in bus_map) & (row['phases'] in {'A', 'B', 'C'}):
        if row['phases']=='A':
            pa = row['kW'] / 1000
            qa = row['kW']/1000 * np.tan(np.arccos(row['PF']))
            
        elif row['phases']=='B':
            pb = row['kW'] / 1000
            qb = row['kW']/1000 * np.tan(np.arccos(row['PF']))
        else: # row['phases']=='C'
            pc = row['kW'] / 1000
            qc = row['kW']/1000 * np.tan(np.arccos(row['PF']))
        load = pp.create_asymmetric_load(
            net, bus=bus_id,
            p_a_mw=pa, q_a_mvar=qa,
            p_b_mw=pb, q_b_mvar=qb,
            p_c_mw=pc, q_c_mvar=qc,
            name=row['Name']
        )
        load_map[load] = row['Yearly']

net.asymmetric_load.head(10)