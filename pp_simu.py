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
    pv_count = len(net.asymmetric_sgen) + 1
    sgen_idx = pp.create_asymmetric_sgen(
        net, bus=bus,
        **p_dict, **q_dict,
        name=f"PV{pv_count}_{bus}{phase.upper()}"
    )

    if ctrl:
        ds = kwargs.get('data_source')
        if ds is None: raise "[MissingDFData] Please provide 'data_source'"

        profile_name = f"CTRL_PV{sgen_idx}_{phase.upper()}"
        ds.df[profile_name] = generate_pv_profile(ds, pv_max_kw=kw)

        pq_area = ppc.controller.DERController.PQVAreas.PQArea4105(variant=1)
        ppc.DERController(
            net=net, element='sgen',
            element_index=sgen_idx,
            pqv_area=pq_area, data_source=ds,
            p_profile=profile_name
        )

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

    # Optional: constant control (if needed for advanced simulations)
    if ctrl:
        ds = kwargs.get('data_source')
        if ds is None: raise 'MissingDFData'

        profile_name = f"CTRL_EV{ev_idx}_p_{phase.lower()}_mw"
        ds.df[profile_name] = generate_pv_profile(ds, ev_max_kw=kw)

        ppc.ConstControl(
            net, element_index=ev_idx,
            element="asymmetric_load",
            profile_name=profile_name, data_source=ds,
            variable=f"p_{phase.lower()}_mw"
        )

    return ev_idx

# ===================================================================
# DETERMINISTIC MODULE
# ===================================================================
def hc_deterministic(net, add_kw=1.0, max_kw=30.0, pv=True, ev=False):
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

    phases = ['A', 'B', 'C']
    hc_results = pd.DataFrame(index=net.bus.index)
    hc_results['bus_name'] = net.bus['name'].values

    for element in elements:
        for phase in phases:
            hc_results[f"{element}_{phase}"] = 0.0

    for bus_idx in net.bus.index[2:]:
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

                    pp.runpp_3ph(net_copy, max_iteration=100, tolerance_mva=1e-6)

                    is_violated, violation = cbn.hc_violation(net_copy, mod='det')
                    if is_violated:
                        print(f'{violation} violation at bus {bus_idx}, phase {p.upper()} with {total_kw} kW')
                        # break
                    else:
                       hc_pv += total_kw if pv else 0
                       hc_ev += total_kw if ev else 0

                except Exception as e:
                    print(f"Stopped at bus {bus_idx}, phase {p.upper()} with {total_kw} kW due to error: {e}")
                    break
                finally:
                    total_kw += add_kw
            hc_results.at[bus_idx, f"PV_{p.upper()}"] = hc_pv
            hc_results.at[bus_idx, f"EV_{p.upper()}"] = hc_ev

    return hc_results



# ===================================================================
# STOCHASTIC MODULE
# ===================================================================
def create_load_controllers(net, ds, **kwargs):
    """
    Assumes that ds.df contains columns named after Loads.xlsx['Name'] and their '_Q' suffix for reactive profiles.
    """
    data_dir = kwargs.get('data_dir')
    data_dir = kwargs.get('data_dir') if data_dir is not None else os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV')
    loads = pd.read_excel(os.path.join(data_dir, "Loads.xlsx"), skiprows=2)

    ppc.ConstControl(
        net, element='asymmetric_load', variable='p_a_mw',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'A'].Name)].index,
        data_source=ds, profile_name=loads[loads['phases']=='A'].Name
    )
    ppc.ConstControl(
        net, element='asymmetric_load', variable='q_a_mvar',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'A'].Name)].index,
        data_source=ds, profile_name=loads[loads['phases']=='A'].Name+'_Q'
    )
    ppc.ConstControl(
        net, element='asymmetric_load', variable='p_b_mw',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'B'].Name)].index,
        data_source=ds, profile_name=loads[loads['phases']=='B'].Name
    )
    ppc.ConstControl(
        net, element='asymmetric_load', variable='q_b_mvar',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'B'].Name)].index, data_source=ds,
        profile_name=loads[loads['phases']=='B'].Name+'_Q'
    )
    ppc.ConstControl(
        net, element='asymmetric_load', variable='p_c_mw',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'C'].Name)].index, 
        data_source=ds, profile_name=loads[loads['phases']=='C'].Name
    )
    ppc.ConstControl(
        net, element='asymmetric_load', variable='q_c_mvar',
        element_index=net.asymmetric_load[net.asymmetric_load.name.isin(loads[loads['phases'] == 'C'].Name)].index, 
        data_source=ds, profile_name=loads[loads['phases']=='C'].Name+'_Q'
    )
    return net

def generate_pv_profile(ds, pv_max_kw=0.5):
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

    profile_dir = kwargs.get('profile_dir')
    profile_dir = os.path.join(os.path.dirname(__file__), 'Modified_116_LV_CSV', 'Load profiles') if profile_dir is None else profile_dir

    file_path = os.path.join(profile_dir, f"Load_profile_{pick_profile}.csv")
    try:
        profile = pd.read_csv(file_path)

    except:
        profile = pd.read_csv(file_path, sep=';')

    finally:
        return profile["mult"].values * ev_max_kw * 1e-3


def hc_montecarlo(net, data_source, output_path, max_iteration=1000, add_kw=1.0, max_kw=30.0, pv=True, ev=False):
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

    for element in elements:
        hc_results[f"{element}_total"] = 0.0

    for bus_idx in net.bus.index[2:6]:
        for i in range(max_iteration):
            net_copy = deepcopy(net)
            create_load_controllers(net_copy, data_source)
            
            total_kw = 0.0
            while total_kw <= max_kw:
                try:
                    phase = choice(phases)
                    der_type = choice(elements)
                    rand_kw = uniform(add_kw, add_kw * 5)
                    time_steps = data_source.df.index

                    if der_type == 'PV':
                        addPV(net_copy, bus_idx, phase, kw=rand_kw, ctrl=True, data_source=data_source)
                    elif der_type == 'EV':
                        addEV(net_copy, bus_idx, phase, kw=rand_kw, ctrl=True, data_source=data_source)

                    # Set up OutputWriter
                    ow = OutputWriter(net_copy, time_steps, output_path=output_path, output_file_type=".csv")
                    ow.log_variable('res_bus_3ph', 'vm_a_pu')
                    ow.log_variable('res_bus_3ph', 'vm_b_pu')
                    ow.log_variable('res_bus_3ph', 'vm_c_pu')
                    ow.log_variable('res_line_3ph', 'loading_a_percent')
                    ow.log_variable('res_line_3ph', 'loading_b_percent')
                    ow.log_variable('res_line_3ph', 'loading_c_percent')
                    ow.log_variable('res_trafo_3ph', 'loading_percent')

                    run_timeseries(net_copy, time_steps=time_steps, run_control=True, continue_on_divergence=False)

                    violated, violation_type = cbn.hc_violation(net_copy, mod='sto')
                    if violated:
                        summary_results.iloc[len(summary_results)] = {
                            'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                            'bus_idx': bus_idx,
                            'installed_kW': total_kw,
                            'violation': violation_type
                        }
                        break
                    else:
                        total_kw += rand_kw

                except Exception as err:
                    summary_results.loc[len(summary_results)] = {
                        'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                        'bus_idx': bus_idx,
                        'installed_kW': total_kw,
                        'violation': str(err)
                    }
                    break

            for element in elements:
                hc_results.at[bus_idx, f"{element}_total"] += total_kw / max_iteration
    summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults.csv"))
    hc_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_HCResults.csv"))
    return hc_results, summary_results