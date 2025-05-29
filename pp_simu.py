import pandapower.control as ctrl
import pandapower as pp
import pandapower.control.characteristic as characteristic

def add_pv_with_voltvar(net, bus, p_kw, phase='A', kvar_margin=0.1):
    """
    Adds a PV unit with Volt-VAR control to a pandapower net.

    Parameters:
        net (pandapowerNet): The network
        bus (int): Bus index where PV is connected
        p_kw (float): Real power in kW
        phase (str): Phase to connect ('A', 'B', 'C')
        kvar_margin (float): Percentage of p_kw reserved for reactive power (e.g., 0.1 = 10%)
    """

    # Convert to MW
    p_mw = p_kw / 1000
    q_max = p_mw * kvar_margin

    # Add PV generator as negative load
    load_args = dict(p_mw=0, q_mvar=0)
    if phase == 'A':
        load_args['p_a_mw'] = -p_mw
        load_args['q_a_mvar'] = 0.0
    elif phase == 'B':
        load_args['p_b_mw'] = -p_mw
        load_args['q_b_mvar'] = 0.0
    elif phase == 'C':
        load_args['p_c_mw'] = -p_mw
        load_args['q_c_mvar'] = 0.0
    else:
        raise ValueError("Phase must be 'A', 'B', or 'C'.")

    load_idx = pp.create_asymmetric_load(net, bus=bus, name="PV_inverter", **load_args)

    # Define Volt-VAR characteristic (simplified IEEE 1547-2018)
    voltage_points = [0.92, 0.97, 1.03, 1.08]
    q_points = [ q_max, 0.0, 0.0, -q_max]
    voltvar_curve = characteristic.Characteristic(voltage_points, q_points, characteristic_name="voltvar")

    # Attach control to the inverter
    if phase == 'A':
        ctrl.ConstControl(net, element='asymmetric_load', variable='q_a_mvar', element_index=[load_idx],
                          data_source=ctrl.CharacteristicControl, profiles={"vm_pu": voltvar_curve}, profile_name="vm_pu")
    elif phase == 'B':
        ctrl.ConstControl(net, element='asymmetric_load', variable='q_b_mvar', element_index=[load_idx],
                          data_source=ctrl.CharacteristicControl, profiles={"vm_pu": voltvar_curve}, profile_name="vm_pu")
    elif phase == 'C':
        ctrl.ConstControl(net, element='asymmetric_load', variable='q_c_mvar', element_index=[load_idx],
                          data_source=ctrl.CharacteristicControl, profiles={"vm_pu": voltvar_curve}, profile_name="vm_pu")
