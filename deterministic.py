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