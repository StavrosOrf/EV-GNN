import numpy as np
from torch_geometric.data import Data
import math


def PublicPST_GNN(env, *args):
    ''' 
    The state function for the GNN model.
    It returns the state of the environment in a format that can be used by the GNN model.    
    '''

    PublicPST_GNN.node_sizes = {'ev': 6, 'cs': 4, 'tr': 2, 'env': 5}

    # create the graph of the environment having as nodes the CPO, the transformers, the charging stations and the EVs connected to the charging stations

    ev_features = []
    cs_features = []
    tr_features = []
    env_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    node_counter = 0

    env_features.append(setpoint)
    env_features.append(env.current_power_usage[env.current_step-1])
    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_counter += 1
    node_names = ['env']

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []  # It is a list that maps the node index to the action index
    # action_mask = [] # It is a list of length equal to the number of nodes, where each element is a list of length equal to the number of actions, with 1 if the action is allowed and 0 otherwise

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    # Map tr.id, cs.id, ev.id to node index
    for cs in env.charging_stations:
        n_ports = cs.n_ports
        for i in range(n_ports):
            mapper[f'Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{i}'] = port_counter + i

        port_counter += n_ports

    for tr in env.transformers:
        # If EV is connected to the charging station that is connected to the transformer
        # Then include transformer id, EV id, EV soc, EV total energy exchanged, EV max charge power, EV min charge power, time of arrival
        registered_tr = False

        for cs in env.charging_stations:
            registered_CS = False

            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:

                        if not registered_CS:
                            registered_CS = True

                            charger_features = [cs.min_charge_current,
                                                cs.max_charge_current,
                                                cs.n_ports,
                                                cs.id
                                                ]

                            if not registered_tr:
                                node_features.append([tr.max_power[env.current_step],
                                                      tr.id
                                                      ])
                                tr_features.append([tr.max_power[env.current_step],
                                                    tr.id
                                                    ])

                                tr_indexes.append(node_counter)
                                node_counter += 1
                                node_types.append(1)
                                node_names.append(f'Tr_{tr.id}')
                                tr_node_index = len(node_names)-1

                                edge_index_from.append(0)
                                edge_index_to.append(tr_node_index)

                                edge_index_from.append(tr_node_index)
                                edge_index_to.append(0)

                                registered_tr = True

                            node_features.append(charger_features)
                            cs_features.append(charger_features)

                            cs_indexes.append(node_counter)
                            node_counter += 1
                            node_types.append(2)
                            node_names.append(f'Tr_{tr.id}_CS_{cs.id}')
                            cs_node_index = len(node_names)-1

                            edge_index_from.append(tr_node_index)
                            edge_index_to.append(cs_node_index)

                            edge_index_from.append(cs_node_index)
                            edge_index_to.append(tr_node_index)

                            registered_CS = True

                        node_features.append([1 if EV.get_soc() == 1 else 0.5,
                                              EV.total_energy_exchanged,
                                              (env.current_step -
                                               EV.time_of_arrival),
                                              EV.id,
                                              cs.id,
                                              tr.id
                                              ])
                        ev_features.append([1 if EV.get_soc() == 1 else 0.5,
                                            EV.total_energy_exchanged,
                                            (env.current_step -
                                             EV.time_of_arrival),
                                            EV.id,
                                            cs.id,
                                            tr.id
                                            ])

                        ev_indexes.append(node_counter)
                        node_counter += 1

                        node_types.append(3)
                        action_mapper.append(
                            mapper[f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}'])
                        node_names.append(f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}')
                        ev_node_index = len(node_names)-1

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(ev_node_index)

                        edge_index_from.append(ev_node_index)
                        edge_index_to.append(cs_node_index)

            # map the edge node names from edge_index_from and edge_index_to to integers

    edge_index = [edge_index_from, edge_index_to]

    data = Data(ev_features=np.array(ev_features).reshape(-1, 6).astype(float),
                cs_features=np.array(cs_features).reshape(-1, 4).astype(float),
                tr_features=np.array(tr_features).reshape(-1, 2).astype(float),
                env_features=np.array(
                    env_features).reshape(-1, 5).astype(float),
                edge_index=np.array(edge_index).astype(int),
                node_types=np.array(node_types).astype(int),
                sample_node_length=[len(node_features)],
                action_mapper=action_mapper,
                ev_indexes=np.array(ev_indexes),
                cs_indexes=np.array(cs_indexes),
                tr_indexes=np.array(tr_indexes),
                env_indexes=np.array(env_indexes),
                )

    return data


def PublicPST_GNN_full_graph(env, *args):
    " This is the state function for the GNN model that uses the full graph representation of the environment without pruning."

    PublicPST_GNN_full_graph.node_sizes = {'ev': 6, 'cs': 4, 'tr': 2, 'env': 5}

    # create the graph of the environment having as nodes the CPO, the transformers, the charging stations and the EVs connected to the charging stations

    ev_features = []
    cs_features = []
    tr_features = []
    env_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    node_counter = 0

    env_features.append(setpoint)
    env_features.append(env.current_power_usage[env.current_step-1])
    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_counter += 1
    node_names = ['env']

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []  # It is a list that maps the node index to the action index
    # action_mask = [] # It is a list of length equal to the number of nodes, where each element is a list of length equal to the number of actions, with 1 if the action is allowed and 0 otherwise

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    # Map tr.id, cs.id, ev.id to node index
    for cs in env.charging_stations:
        n_ports = cs.n_ports
        for i in range(n_ports):
            mapper[f'Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{i}'] = port_counter + i

        port_counter += n_ports

    for tr in env.transformers:

        registered_tr = False

        for cs in env.charging_stations:
            registered_CS = False

            if cs.connected_transformer == tr.id:
                for port_n in range(cs.n_ports):
                    if env.current_step == env.simulation_length:
                        continue
                    
                    EV = cs.evs_connected[i]

                    if not registered_CS:
                        registered_CS = True

                        charger_features = [cs.min_charge_current,
                                            cs.max_charge_current,
                                            cs.n_ports,
                                            cs.id
                                            ]

                        if not registered_tr:
                            node_features.append([tr.max_power[env.current_step],
                                                  tr.id
                                                  ])
                            tr_features.append([tr.max_power[env.current_step],
                                                tr.id
                                                ])

                            tr_indexes.append(node_counter)
                            node_counter += 1
                            node_types.append(1)
                            node_names.append(f'Tr_{tr.id}')
                            tr_node_index = len(node_names)-1

                            edge_index_from.append(0)
                            edge_index_to.append(tr_node_index)

                            edge_index_from.append(tr_node_index)
                            edge_index_to.append(0)

                            registered_tr = True

                        node_features.append(charger_features)
                        cs_features.append(charger_features)

                        cs_indexes.append(node_counter)
                        node_counter += 1
                        node_types.append(2)
                        node_names.append(f'Tr_{tr.id}_CS_{cs.id}')
                        cs_node_index = len(node_names)-1

                        edge_index_from.append(tr_node_index)
                        edge_index_to.append(cs_node_index)

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(tr_node_index)

                        registered_CS = True

                        if EV is not None:
                            node_features.append([1 if EV.get_soc() == 1 else 0.5,
                                                  EV.total_energy_exchanged,
                                                  (env.current_step -
                                                 EV.time_of_arrival),
                                                  EV.id,
                                                  cs.id,
                                                  tr.id
                                                  ])
                            ev_features.append([1 if EV.get_soc() == 1 else 0.5,
                                                EV.total_energy_exchanged,
                                                (env.current_step -
                                                EV.time_of_arrival),
                                                EV.id,
                                                cs.id,
                                                tr.id
                                                ])
                            
                            action_mapper.append(
                                mapper[f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}'])

                        else:
                            node_features.append([0, 0, 0, 0, cs.id, tr.id])
                            ev_features.append([0, 0, 0, 0, cs.id, tr.id])
                            
                            
                        node_names.append(
                            f'Tr_{tr.id}_CS_{cs.id}_EV_{port_n}')

                        ev_indexes.append(node_counter)
                        node_counter += 1

                        node_types.append(3)

                        ev_node_index = len(node_names)-1

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(ev_node_index)

                        edge_index_from.append(ev_node_index)
                        edge_index_to.append(cs_node_index)

            # map the edge node names from edge_index_from and edge_index_to to integers

    edge_index = [edge_index_from, edge_index_to]

    data = Data(ev_features=np.array(ev_features).reshape(-1, 6).astype(float),
                cs_features=np.array(cs_features).reshape(-1, 4).astype(float),
                tr_features=np.array(tr_features).reshape(-1, 2).astype(float),
                env_features=np.array(
                    env_features).reshape(-1, 5).astype(float),
                edge_index=np.array(edge_index).astype(int),
                node_types=np.array(node_types).astype(int),
                sample_node_length=[len(node_features)],
                action_mapper=action_mapper,
                ev_indexes=np.array(ev_indexes),
                cs_indexes=np.array(cs_indexes),
                tr_indexes=np.array(tr_indexes),
                env_indexes=np.array(env_indexes),
                )

    return data


def PublicPST_GNN_no_position_encoding(env, *args):
    ''' 
    The state function for the GNN model without the position encoding.
    It returns the state of the environment in a format that can be used by the GNN model.    
    '''
    PublicPST_GNN_no_position_encoding.node_sizes = {
        'ev': 3, 'cs': 3, 'tr': 1, 'env': 5}

    # create the graph of the environment having as nodes the CPO, the transformers, the charging stations and the EVs connected to the charging stations

    ev_features = []
    cs_features = []
    tr_features = []
    env_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    node_counter = 0

    env_features.append(setpoint)
    env_features.append(env.current_power_usage[env.current_step-1])
    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_counter += 1
    node_names = ['env']

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []  # It is a list that maps the node index to the action index
    # action_mask = [] # It is a list of length equal to the number of nodes, where each element is a list of length equal to the number of actions, with 1 if the action is allowed and 0 otherwise

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    # Map tr.id, cs.id, ev.id to node index
    for cs in env.charging_stations:
        n_ports = cs.n_ports
        for i in range(n_ports):
            mapper[f'Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{i}'] = port_counter + i

        port_counter += n_ports

    for tr in env.transformers:
        # If EV is connected to the charging station that is connected to the transformer
        # Then include transformer id, EV id, EV soc, EV total energy exchanged, EV max charge power, EV min charge power, time of arrival
        registered_tr = False

        for cs in env.charging_stations:
            registered_CS = False

            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:

                        if not registered_CS:
                            registered_CS = True

                            charger_features = [cs.min_charge_current,
                                                cs.max_charge_current,
                                                cs.n_ports,
                                                ]

                            if not registered_tr:
                                node_features.append([tr.max_power[env.current_step],
                                                      ])
                                tr_features.append([tr.max_power[env.current_step],
                                                    ])

                                tr_indexes.append(node_counter)
                                node_counter += 1
                                node_types.append(1)
                                node_names.append(f'Tr_{tr.id}')
                                tr_node_index = len(node_names)-1

                                edge_index_from.append(0)
                                edge_index_to.append(tr_node_index)

                                edge_index_from.append(tr_node_index)
                                edge_index_to.append(0)

                                registered_tr = True

                            node_features.append(charger_features)
                            cs_features.append(charger_features)

                            cs_indexes.append(node_counter)
                            node_counter += 1
                            node_types.append(2)
                            node_names.append(f'Tr_{tr.id}_CS_{cs.id}')
                            cs_node_index = len(node_names)-1

                            edge_index_from.append(tr_node_index)
                            edge_index_to.append(cs_node_index)

                            edge_index_from.append(cs_node_index)
                            edge_index_to.append(tr_node_index)

                            registered_CS = True

                        node_features.append([1 if EV.get_soc() == 1 else 0.5,
                                              EV.total_energy_exchanged,
                                              (env.current_step -
                                               EV.time_of_arrival),
                                              ])
                        ev_features.append([1 if EV.get_soc() == 1 else 0.5,
                                            EV.total_energy_exchanged,
                                            (env.current_step -
                                             EV.time_of_arrival),
                                            ])

                        ev_indexes.append(node_counter)
                        node_counter += 1

                        node_types.append(3)
                        action_mapper.append(
                            mapper[f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}'])
                        node_names.append(f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}')
                        ev_node_index = len(node_names)-1

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(ev_node_index)

                        edge_index_from.append(ev_node_index)
                        edge_index_to.append(cs_node_index)

            # map the edge node names from edge_index_from and edge_index_to to integers

    edge_index = [edge_index_from, edge_index_to]

    data = Data(ev_features=np.array(ev_features).reshape(-1, 3).astype(float),
                cs_features=np.array(cs_features).reshape(-1, 3).astype(float),
                tr_features=np.array(tr_features).reshape(-1, 1).astype(float),
                env_features=np.array(
                    env_features).reshape(-1, 5).astype(float),
                edge_index=np.array(edge_index).astype(int),
                node_types=np.array(node_types).astype(int),
                sample_node_length=[len(node_features)],
                action_mapper=action_mapper,
                ev_indexes=np.array(ev_indexes),
                cs_indexes=np.array(cs_indexes),
                tr_indexes=np.array(tr_indexes),
                env_indexes=np.array(env_indexes),
                )

    return data


def V2G_ProfitMax_with_Loads_GNN(env, *args):
    ''' 
    The state function of the profit maximization model with V2G capabilities for the GNN models.
    '''

    V2G_ProfitMax_with_Loads_GNN.node_sizes = {
        'ev': 5, 'cs': 4, 'tr': 2, 'env': 5}

    # create the graph of the environment having as nodes the CPO, the transformers, the charging stations and the EVs connected to the charging stations

    ev_features = []
    cs_features = []
    tr_features = []
    env_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    node_counter = 0

    if env.current_step < env.simulation_length:
        env_features.append(abs(env.charge_prices[0, env.current_step]))
    else:
        env_features.append(0)

    env_features.append(env.current_power_usage[env.current_step-1])
    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_counter += 1
    node_names = ['env']

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []  # It is a list that maps the node index to the action index

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    # Map tr.id, cs.id, ev.id to node index
    for cs in env.charging_stations:
        n_ports = cs.n_ports
        for i in range(n_ports):
            mapper[f'Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{i}'] = port_counter + i

        port_counter += n_ports

    for tr in env.transformers:
        # If EV is connected to the charging station that is connected to the transformer
        # Then include transformer id, EV id, EV soc, EV total energy exchanged, EV max charge power, EV min charge power, time of arrival
        registered_tr = False

        for cs in env.charging_stations:
            registered_CS = False

            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:

                        if not registered_CS:
                            registered_CS = True

                            charger_features = [cs.min_charge_current,
                                                cs.max_charge_current,
                                                cs.n_ports,
                                                cs.id
                                                ]

                            if not registered_tr:
                                node_features.append([tr.max_power[env.current_step] -
                                                      tr.inflexible_load[env.current_step] +
                                                      tr.solar_power[env.current_step],
                                                      tr.id
                                                      ])
                                tr_features.append([tr.max_power[env.current_step] -
                                                    tr.inflexible_load[env.current_step] +
                                                    tr.solar_power[env.current_step],
                                                    tr.id
                                                    ])

                                tr_indexes.append(node_counter)
                                node_counter += 1
                                node_types.append(1)
                                node_names.append(f'Tr_{tr.id}')
                                tr_node_index = len(node_names)-1

                                edge_index_from.append(0)
                                edge_index_to.append(tr_node_index)

                                edge_index_from.append(tr_node_index)
                                edge_index_to.append(0)

                                registered_tr = True

                            node_features.append(charger_features)
                            cs_features.append(charger_features)

                            cs_indexes.append(node_counter)
                            node_counter += 1
                            node_types.append(2)
                            node_names.append(f'Tr_{tr.id}_CS_{cs.id}')
                            cs_node_index = len(node_names)-1

                            edge_index_from.append(tr_node_index)
                            edge_index_to.append(cs_node_index)

                            edge_index_from.append(cs_node_index)
                            edge_index_to.append(tr_node_index)

                            registered_CS = True

                        node_features.append([EV.get_soc(),
                                              EV.time_of_departure - env.current_step,
                                              EV.id,
                                              cs.id,
                                              tr.id
                                              ])
                        ev_features.append([EV.get_soc(),
                                            EV.time_of_departure - env.current_step,
                                            EV.id,
                                            cs.id,
                                            tr.id
                                            ])

                        ev_indexes.append(node_counter)
                        node_counter += 1

                        node_types.append(3)
                        action_mapper.append(
                            mapper[f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}'])
                        node_names.append(f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}')
                        ev_node_index = len(node_names)-1

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(ev_node_index)

                        edge_index_from.append(ev_node_index)
                        edge_index_to.append(cs_node_index)

            # map the edge node names from edge_index_from and edge_index_to to integers

    edge_index = [edge_index_from, edge_index_to]

    data = Data(ev_features=np.array(ev_features).reshape(-1, 5).astype(float),
                cs_features=np.array(cs_features).reshape(-1, 4).astype(float),
                tr_features=np.array(tr_features).reshape(-1, 2).astype(float),
                env_features=np.array(
                    env_features).reshape(-1, 5).astype(float),
                edge_index=np.array(edge_index).astype(int),
                node_types=np.array(node_types).astype(int),
                sample_node_length=[len(node_features)],
                action_mapper=action_mapper,
                ev_indexes=np.array(ev_indexes),
                cs_indexes=np.array(cs_indexes),
                tr_indexes=np.array(tr_indexes),
                env_indexes=np.array(env_indexes),
                )

    return data
