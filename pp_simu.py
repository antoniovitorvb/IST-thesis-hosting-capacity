import os
import pandapower.control as ppc
import pandapower as pp
import create_basic_network as cbn
import numpy as np
import pandas as pd
from copy import deepcopy

def addPV(net, bus, phase, kw=1.0, ctrl=False, **kwargs):
    """
    Adds an asymmetric PV system to the network with a DERController for Volt-VAR control.
    
    Parameters:
    - net: pandapower network
    - bus: int, the bus index
    - phase: str, one of 'a', 'b', 'c'
    - kw: float, power injected in kW (default 1.0)
    """
    p_mw = kw / 1000  # Convert to MW

    # Phase-specific power injection
    p_dict = {'p_a_mw': 0.0, 'p_b_mw': 0.0, 'p_c_mw': 0.0}
    q_dict = {'q_a_mw': 0.0, 'q_b_mw': 0.0, 'q_c_mw': 0.0}
    
    if phase.lower() in ['a', 'b', 'c']: p_dict[f"p_{phase.lower()}_mw"] = p_mw
    else: raise ValueError("Phase must be 'A', 'B', or 'C'.")

    # Add the asymmetric PV generator
    pv_count = len(net.asymmetric_sgen)
    sgen_idx = pp.create_asymmetric_sgen(
        net, bus=bus,
        **p_dict, **q_dict,
        name=f"PV{pv_count}_{bus}{phase.upper()}"
    )
    # print(f"{len(net.asymmetric_sgen)} PV Gen created so far...")

    if ctrl:
        ds = kwargs.get('data_source')
        if ds is None: raise Exception("[MissingDFData] Please provide 'data_source'")

        profile_name = f"CTRL_PV{sgen_idx}_{phase.upper()}"
        ds.df[profile_name] = generate_pv_profile(ds, pv_max_kw=kw)

        ppc.ConstControl(
            net=net, element='asymmetric_sgen',
            element_index=sgen_idx, data_source=ds,
            profile_name=profile_name,
            variable=f"p_{phase.lower()}_mw"
        )
        # print(f"Created {profile_name}!")

        # pq_area = ppc.controller.DERController.PQVAreas.PQArea4105(variant=1)
        # ppc.DERController(
        #     net=net, element='sgen',
        #     element_index=sgen_idx,
        #     pqv_area=pq_area, data_source=ds,
        #     p_profile=profile_name
        # )
        # print(f"Created {profile_name}!")

    return sgen_idx

def addEV(net, bus, phase, kw=7.0, ctrl=False, **kwargs):
    """
    Adds an asymmetric load representing an EV charger to the network on a specific phase.

    Parameters:
        net (pandapowerNet): The pandapower network.
        bus_idx (int): Index of the bus where the EV charger is connected.
        phase (str): One of 'a', 'b', or 'c'.
        kw (float): Real power in kW (default is 1.0 kW).
    """
    p_mw = kw / 1000  # Convert to MW

    # Phase-specific power injection
    p_dict = {'p_a_mw': 0.0, 'p_b_mw': 0.0, 'p_c_mw': 0.0}
    q_dict = {'q_a_mw': 0.0, 'q_b_mw': 0.0, 'q_c_mw': 0.0}
    
    if phase.lower() in ['a', 'b', 'c']: p_dict[f"p_{phase.lower()}_mw"] = p_mw
    else: raise ValueError("Phase must be 'A', 'B', or 'C'.")
    ev_count = net.asymmetric_load['name'].str.contains('EV').sum()
    ev_idx = pp.create_asymmetric_load(
        net, bus=bus,
        **p_dict, **q_dict,
        name=f"EV{ev_count}_{bus}{phase.upper()}"
    )
    # print(f"{net.asymmetric_load['name'].str.contains('EV').sum()} EVs created so far...")

    # Optional: constant control (if needed for advanced simulations)
    if ctrl:
        ds = kwargs.get('data_source')
        if ds is None: raise Exception("[MissingDFData] Please provide 'data_source'")

        profile_name = f"CTRL_EV{ev_idx}_p_{phase.lower()}_mw"
        ds.df[profile_name] = generate_ev_profile(ds, ev_max_kw=kw)

        ppc.ConstControl(
            net, element_index=ev_idx,
            element="asymmetric_load",
            profile_name=profile_name, data_source=ds,
            variable=f"p_{phase.lower()}_mw"
        )
        # print(f"Created {profile_name}!")

    return ev_idx

# ===================================================================
# DETERMINISTIC MODULE
# ===================================================================
def hc_deterministic(net, add_kw=1.0, max_kw=30.0, pv=True, ev=False, **kwargs):
    """
    Increases DERs until a violation occurs at any bus.

    Parameters:
    - net: pandapower network
    - buses: list of int, bus indices to install DERs
    - phase: str, 'a', 'b', or 'c'
    - step_kw: float, step size in kW
    - max_kw: float, max per-device injection
    - device: 'pv' or 'ev'

    Returns:
    - total_kw: total DER injection before first violation
    """
    elements = []
    if pv: elements.append('PV')
    if ev: elements.append('EV')

    init = kwargs.get('init', 'auto')
    max_iteration = kwargs.get('max_iteration', 100)
    tolerance_mva = kwargs.get('tolerance_mva', 1e-8)
    run_control = kwargs.get('run_control', True)

    output_path = kwargs.get('output_path', os.path.join(os.path.dirname(__file__), 'hc_results', 'DET'))

    phases = ['A', 'B', 'C']
    hc_results = pd.DataFrame(index=net.bus.index)
    hc_results['bus_name'] = net.bus['name'].values
    summary_results = pd.DataFrame(columns=['scenario', 'bus_idx', 'installed_kW', 'violation'])
    

    hc_indices = kwargs.get('hc_indices', net.bus.index[2:])
    bus_indices = net.bus[net.bus.name.isin(hc_indices)].index
    # line_bus_indices = net.line[net.line.to_bus.isin(hc_indices)].index

    for element in elements:
        for phase in phases:
            hc_results[f"{element}_{phase}_total"] = 0.0

    for bus_idx in hc_indices:
        for p in phases:
            net_copy = deepcopy(net)

            # if cbn.hc_violation(net_copy, mod='det'):
            #     hc_results[(bus, p)] = 0.0
            #     continue

            total_kw = hc_pv = hc_ev = 0.0
            while total_kw <= max_kw:
                try:
                    if pv: addPV(net_copy, bus_idx, p, kw=add_kw)
                    if ev: addEV(net_copy, bus_idx, p, kw=add_kw)

                    pp.runpp_3ph(net_copy, init=init, max_iteration=max_iteration, tolerance_mva=tolerance_mva)

                    is_violated, violation = cbn.hc_violation(net_copy, mod='det')
                    if is_violated:
                        print(f'{violation} violation at bus {bus_idx}, phase {p.upper()} with {total_kw} kW')
                        summary_results.loc[len(summary_results)] = {
                            'scenario': f"{''.join(elements)}_bus_{bus_idx}_{p.upper()}",
                            'bus_idx': bus_idx,
                            'installed_kW': total_kw,
                            'violation': violation
                        }
                        # break
                    else:
                       hc_pv += total_kw if pv else 0
                       hc_ev += total_kw if ev else 0

                except Exception as err:
                    print(f"Stopped at bus {bus_idx}, phase {p.upper()} with {total_kw} kW due to error: {err}")
                    summary_results.loc[len(summary_results)] = {
                        'scenario': f"{''.join(elements)}_bus_{bus_idx}_{p.upper()}",
                        'bus_idx': bus_idx,
                        'installed_kW': total_kw,
                        'violation': str(err)
                    }
                    break
                finally:
                    total_kw += add_kw
            hc_results.at[bus_idx, f"PV_{p.upper()}"] = hc_pv
            hc_results.at[bus_idx, f"EV_{p.upper()}"] = hc_ev
    summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults.csv"))
    hc_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_HCResults.csv"))
    return hc_results, summary_results



# ===================================================================
# STOCHASTIC MODULE
# ===================================================================
def create_load_controllers(net, ds, **kwargs):
    """
    Assumes that ds.df contains columns named after Loads.xlsx['Name'] and their '_Q' suffix for reactive profiles.
    """
    data_dir = kwargs.get('data_dir', os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV'))
    loads = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)

    # Ensure load names are strings and match exactly
    loads['Name'] = loads['Name'].astype(str)
    net.asymmetric_load['name'] = net.asymmetric_load['name'].astype(str)
    # print('Creating Controllers')
    # Filter loads by phase
    for phase in ['A', 'B', 'C']:
        
        phase_loads = loads[loads['phases'] == phase]
        matching = net.asymmetric_load[net.asymmetric_load['name'].isin(phase_loads['Name'])]
        # print(f"Phase {phase}: {' '.join([str(i) for i in matching.index])}")
        # Assign control for each power type
        ppc.ConstControl(
            net, element='asymmetric_load', variable=f'p_{phase.lower()}_mw',
            element_index=matching.index.tolist(), data_source=ds, profile_name=phase_loads['Name'].tolist()
        )
        ppc.ConstControl(
            net, element='asymmetric_load', variable=f'q_{phase.lower()}_mvar',
            element_index=matching.index.tolist(), data_source=ds, profile_name=(phase_loads['Name']+'_Q').tolist()
        )

    return net

def generate_pv_profile(ds, pv_max_kw=0.5):
    # print(f"Creating a PV Gen with {pv_max_kw} kW")
    minutes = len(ds.df)
    t = np.arange(0, minutes)

    # Cosine-based profile peaking at noon and bounded by daylight hours
    pv_base = 0.9 * pv_max_kw * np.cos(np.pi * 2 * t / 1440 + np.pi)
    pv_base = np.clip(pv_base, 0, None)

    noise = np.random.normal(0, 0.5, size=minutes)
    pv_noise = pd.Series(pv_base) * (1 + pd.Series(noise).rolling(15, center=True).mean().fillna(0))
    pv_profile = np.clip(pv_noise, 0, pv_max_kw) * 1e-3  # convert to MW

    return pv_profile

def generate_ev_profile(ds, ev_max_kw=7.0, **kwargs):
    minutes = len(ds.df)
    pick_profile = int(np.random.choice(range(56,101)))

    profile_dir = kwargs.get('profile_dir', os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV', 'Load profiles'))

    file_path = os.path.join(profile_dir, f"Load_profile_{pick_profile}.csv")
    try:
        profile = pd.read_csv(file_path)

    except:
        profile = pd.read_csv(file_path, sep=';')

    finally:
        # print(f"Profile shape: {profile.shape} from Load_profile_{pick_profile}")
        return profile["mult"].values * ev_max_kw * 1e-3


def hc_montecarlo(net, data_source, output_path, max_iteration=1000, add_kw=1.0, max_kw=30.0, pv=True, ev=False, **kwargs):
    """
    Run Monte Carlo simulations to assess probabilistic hosting capacity.

    Parameters:
    - net: pandapower network object
    - data_source: DFData object with time-series profiles
    - time_steps: list or index of time steps for QSTS (e.g., ds.df.index)
    - iteration: number of Monte Carlo scenarios
    - add_kw: step size in kW for each DER addition
    - max_kw: maximum DER capacity per bus/phase
    - pv: include PV systems
    - ev: include EV chargers

    Returns:
    - hc_results: DataFrame with estimated hosting capacity per bus
    - summary_results: DataFrame logging violations per scenario
    """
    from random import choice, uniform
    from pandapower.timeseries import run_timeseries, OutputWriter

    elements = []
    if pv: elements.append('PV')
    if ev: elements.append('EV')

    phases = ['A', 'B', 'C']
    hc_results = pd.DataFrame(index=net.bus.index)
    hc_results['bus_name'] = net.bus['name'].values
    summary_results = pd.DataFrame(columns=['scenario', 'bus_idx', 'installed_kW', 'violation'])
    temp_summary_results = pd.DataFrame(columns=['scenario', 'bus_idx', 'phase', 'installed_kW', 'violation'])

    indices = kwargs.get('ow_index', net.bus.index[2:])
    bus_indices = net.bus[net.bus.name.isin(indices)].index
    line_bus_indices = net.line[net.line.to_bus.isin(indices)].index

    for element in elements:
        hc_results[f"{element}_total"] = 0.0

    for bus_idx in bus_indices:
        # print(f"Bus {bus_idx} - ite {i}")
        for i in range(max_iteration):
            print(f"\n\n\n\n\nProgress: {i} / {max_iteration}")
            net_copy = deepcopy(net)
            create_load_controllers(net_copy, data_source)
            
            total_kw = 0.0
            while total_kw <= max_kw:
                try:
                    phase = choice(phases)
                    der_type = choice(elements)
                    pv_rand_kw = ev_rand_kw = 0
                    time_steps = data_source.df.index

                    if der_type == 'PV':
                        pv_rand_kw = uniform(add_kw, add_kw * 5)
                        addPV(net_copy, bus_idx, phase, kw=pv_rand_kw, ctrl=True, data_source=data_source)
                    if der_type == 'EV':
                        ev_rand_kw = uniform(add_kw, add_kw * 5)
                        addEV(net_copy, bus_idx, phase, kw=ev_rand_kw, ctrl=True, data_source=data_source)
                    
                    total_kw += pv_rand_kw + ev_rand_kw

                except Exception as err:
                    print(err)

                try:
                    # Set up OutputWriter
                    # print("Set up OutputWriter...")
                    ow = OutputWriter(net_copy, time_steps, output_path=output_path, output_file_type=".csv")
                    ow.log_variable('res_bus_3ph', 'vm_a_pu', index=bus_indices)
                    ow.log_variable('res_bus_3ph', 'vm_b_pu', index=bus_indices)
                    ow.log_variable('res_bus_3ph', 'vm_c_pu', index=bus_indices)
                    ow.log_variable('res_line_3ph', 'loading_a_percent', index=line_bus_indices)
                    ow.log_variable('res_line_3ph', 'loading_b_percent', index=line_bus_indices)
                    ow.log_variable('res_line_3ph', 'loading_c_percent', index=line_bus_indices)
                    ow.log_variable('res_trafo_3ph', 'loading_percent', index=net.trafo.index)

                    ow.remove_log_variable('res_bus', 'vm_pu')
                    ow.remove_log_variable('res_line', 'loading_percent')

                    print(f"\nBus {bus_idx}-ite {i} Running Time Series\n")
                    run_timeseries(net_copy, time_steps=time_steps, run=pp.runpp_3ph, run_control=True, continue_on_divergence=True)

                    violated, violation_type = cbn.hc_violation(net_copy, mod='sto', output_writer_data=ow.output)
                    if violated:
                        print(f"Violation: {violation_type}")
                        summary_results.loc[len(summary_results)] = {
                            'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                            'bus_idx': bus_idx,
                            'installed_kW': total_kw,
                            'violation': violation_type
                        }
                        break
                    else:
                        pass
                        # print(f"Bus {bus_idx} - ite {i} Deu bom! Total: {total_kw} kW")

                except Exception as err:
                    print(err)
                    summary_results.loc[len(summary_results)] = {
                        'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                        'bus_idx': bus_idx,
                        'installed_kW': total_kw,
                        'violation': str(err)
                    }
                    break
                finally:
                    
                    temp_summary_results.loc[len(temp_summary_results)] = {
                        'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                        'bus_idx': bus_idx,
                        'phase': phase,
                        'installed_kW': total_kw,
                        'violation': violation_type
                    }
                    temp_summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults_BUS{bus_idx}.csv"))

            for element in elements:
                hc_results.at[bus_idx, f"{element}_total"] += total_kw / max_iteration
    summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults.csv"))
    hc_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_HCResults.csv"))
    return hc_results, summary_results