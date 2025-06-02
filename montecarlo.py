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
    temp_summary_results = pd.DataFrame(columns=['scenario', 'bus_idx', 'installed_kW', 'violation'])

    indices = kwargs.get('ow_index', net.bus.index[2:])
    bus_indices = net.bus[net.bus.name.isin(indices)].index
    line_bus_indices = net.line[net.line.to_bus.isin(indices)].index

    for element in elements:
        hc_results[f"{element}_total"] = 0.0

    for i in range(max_iteration):
        for bus_idx in indices:
            # print(f"Bus {bus_idx} - ite {i}")
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

                    print(f"\n\n\n\n\nBus {bus_idx}-ite {i} Running Time Series\n")
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
                        total_kw += rand_kw
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
                    temp_summary_results.loc[len(summary_results)] = {
                        'scenario': f"{''.join(elements)}_bus_{bus_idx}_iter_{i}",
                        'bus_idx': bus_idx,
                        'installed_kW': rand_kw,
                        'violation': violation_type
                    }
                    temp_summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults_{i}.csv"))

            for element in elements:
                hc_results.at[bus_idx, f"{element}_total"] += total_kw / max_iteration
    summary_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_summaryResults.csv"))
    hc_results.to_csv(os.path.join(output_path, f"{''.join(elements)}_HCResults.csv"))
    return hc_results, summary_results