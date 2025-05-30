import pandapower.control as ppc
import pandapower as pp
import create_basic_network as cbn
import numpy as np
import pandas as pd
from copy import deepcopy

def addPV_det(net, bus, phase, kw=1.0, ctrl=False):
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
        pq_area = ppc.controller.DERController.PQVAreas.PQArea4105(variant=1)
        ppc.DERController(
            net=net, element='sgen',
            element_index=sgen_idx,
            pqv_area=pq_area,
            p_profile=f"CTRL_PV{sgen_idx}_{phase.upper()}"
        )

    return sgen_idx

def addEV_det(net, bus, phase, kw=7.0, ctrl=False):
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
        name=f"EV{ev_count}_{bus}{phase.upper}"
    )

    # Optional: constant control (if needed for advanced simulations)
    if ctrl:
        ppc.ConstControl(
            net, element_index=ev_idx,
            element=f"asymmetric_load",
            profile_name=f"CTRL_EV{ev_idx}_p_{phase.lower()}_mw",
            variable=f"p_{phase.lower()}_mw"
        )

    return ev_idx

def hc_deterministic(net, add_kw=1.0, max_kw=30.0, pv=True, ev=True):
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
    phases = ['A', 'B', 'C']
    hc_results =pd.DataFrame(index=net.bus.index)
    hc_results['bus_name'] = net.bus['name'].values
    for phase in phases:
        hc_results[phase.upper()] = 0.0

    for bus_idx in net.bus.index[2:]:
        for p in phases:
            net_copy = deepcopy(net)

            # if cbn.hc_violation(net_copy, mod='det'):
            #     hc_results[(bus, p)] = 0.0
            #     continue

            total_kw = hc_kw = 0.0
            while total_kw <= max_kw:
                try:
                    if pv: addPV_det(net_copy, bus_idx, p, kw=add_kw)
                    if ev: addEV_det(net_copy, bus_idx, p, kw=add_kw)

                    pp.runpp_3ph(net_copy, max_iteration=100, tolerance_mva=1e-6)

                    is_violated, violation = cbn.hc_violation(net_copy, mod='det')
                    if is_violated:
                        print(f'{violation} violation at bus {bus_idx}, phase {p.upper()} with {total_kw} kW')
                        # break
                    else:
                        hc_kw = total_kw

                    total_kw += add_kw
                except Exception as e:
                    print(f"Stopped at bus {bus_idx}, phase {p.upper()} with {total_kw} kW due to error: {e}")
                    # break
                finally:
                    total_kw += add_kw
            
            hc_results.at[bus_idx, p.upper()] = hc_kw

    return hc_results