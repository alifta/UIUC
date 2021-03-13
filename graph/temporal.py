# ------
# Import
# ------

import networkx as nx

from collections import Counter
from collections import defaultdict
from itertools import permutations
from networkx.algorithms.core import onion_layers

import powerlaw

from sklearn.preprocessing import QuantileTransformer

from .io import *
from .utils import *

# ----------------
# Temporal Network
# ----------------


def temporal_bt(
    folder_in=[FILE, DB],
    folder_out=[NETWORK],
    file_in=['bt.csv', 'uiuc.db'],
    file_out=[
        'bt_temporal_network.gpickle',
        'bt_bipartite_network.gpickle',
        'bt_temporal_edgelist.csv',
        'bt_bipartite_edgelist.csv',
        'bt_temporal_times.csv',
        'bt_bipartite_times.csv',
        'bt_temporal_nodes.csv',
        'bt_bipartite_nodes.csv',
        'bt_temporal_weights.csv',
        'bt_bipartite_weights.csv',
        'bt_temporal_weights_scaled.csv',
        'bt_bipartite_weights_scaled.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    output_times=False,
    output_network=True,
    plot_weights=False,
    save_times=True,
    save_nodes=True,
    save_weights=True,
    save_network_db=True,
    save_network_csv=True,
    save_network_file=True
):
    """
    Read dataset CSV file that has (at least) 3 columns of (node-1, node-2, timestamp)
    and create temporal network which is a multiple-edge directed (aggregated) temporal graph
    with time-labeled edges of (u,v,t)
    """

    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    path2 = path_edit(
        [file_in[1]],
        folder_in[1],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # G is user-user network
    graph = nx.MultiDiGraph()
    graph.name = 'Tremporal Network'
    # B is user-other-devices biparite network
    bipartite = nx.MultiDiGraph()
    bipartite.name = 'Bipartite Tremporal Network'

    # Read dataset
    data = pd.read_csv(
        path1,
        header=None,
        names=['user', 'mac', 'time'],
        parse_dates=['time']
    )

    # Print times distributions
    if output_times:
        times = pd.Series(sorted(data.time.unique()))
        print(f'Number of unique timestamps = {len(times)}')
        print('Timestamp : Frequency')
        for t_size in data.groupby('time').size().iteritems():
            print(
                f'{times[times == t_size[0]].index[0]}) {t_size[0]} : {t_size[1]}'
            )
        print()

    # Create timestamp list
    times = []
    times_bipartite = []

    # Dictionary {time:{(user-1,user-2):weight}}
    time_edges = defaultdict()  # User -> User
    time_bipartite_edges = defaultdict()  # Users -> Others

    # Group interactions by time filtering
    for key_time, group_user_mac in data[['user',
                                          'mac']].groupby(data['time']):
        # Co-location graph edges in filtered timestamp
        temp_edges = []
        for key_mac, group_connection in group_user_mac.groupby(['mac']
                                                                )['user']:
            # Users of connecting to filtered MAC
            temp_users = list(group_connection.unique())
            # If the ID of shared connected mac is [0-27]
            # Users directly connect to another bluetooth user
            if key_mac < 28:
                # Case where only one direct connection: user -> user
                if len(temp_users) <= 1:
                    temp_edges.append((temp_users[0], key_mac))
                    # Comment next line, if wanna have directed edges
                    temp_edges.append((key_mac, temp_users[0]))
                else:
                    # Case where more than 1 user undirectly connect together via 1 direct user -> user edge
                    # Direct edges
                    for element in temp_users:
                        temp_edges.append((element, key_mac))
                        # Uncomment next line, if wanna have undirected edges when one user observe another user directly
                        temp_edges.append((key_mac, element))
                    # Undirect edges
                    connected_users = list(permutations(temp_users, 2))
                    connected_users = [tuple(e) for e in connected_users]
                    temp_edges.extend(connected_users)
            # If users are connected to device with ID > 28
            # Meaning indirect edges with each other
            else:
                # Only consider cases of more than 1 unique user for co-location indirected edges
                if len(temp_users) > 1:
                    # Undirect edges
                    connected_users = list(permutations(temp_users, 2))
                    connected_users = [tuple(e) for e in connected_users]
                    temp_edges.extend(connected_users)
        # Add edges of current timestamp (with their strength) to dictionary
        if len(temp_edges) > 0:
            time_edges[key_time] = dict(Counter(temp_edges))
        # Bipartite graph edges
        # We don't care about MAC < 28, just want to count
        # How many times in each timestamp a user connect to a MAC
        # Dictionary {time:{(user,mac):weight}}
        bipartite_edges = {}
        # Filter connections based on (user -> mac) interaction and its weight
        for key_mac, group_connection in group_user_mac.groupby(
            ['mac', 'user']
        ):
            # User connected to filtered MAC with X number of times
            bipartite_edges[key_mac] = len(group_connection)
        # Add edges of this time (with their strength) to dictionary
        time_bipartite_edges[key_time] = bipartite_edges

    # Co-location network data
    l1, l2, l3, l4 = [], [], [], []  # time, node, node, weight
    for k1, v1 in time_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_graph = pd.DataFrame(
        list(zip(l1, l2, l3, l4)), columns=['t', 'u', 'v', 'w']
    )

    # Scale edge weights to range [0-1]
    X = [[entry] for entry in data_graph['w']]
    if save_weights: np.savetxt(file_out[8], X, delimiter=',', fmt='%s')

    # Plot the distribution of original weights
    if plot_weights:
        plt.figure()
        ax = sns.distplot(
            X, bins=max(X), kde=True, hist_kws={
                "linewidth": 15,
                'alpha': 1
            }
        )
        ax.set(xlabel='Original Edge Weight', ylabel='Frequency')

    # Max-Min Normalizer (produce many zeros)
    # transformer = MinMaxScaler()
    # X_scaled = transformer.fit_transform(X)
    # Returning column to row vector again
    # X_scaled = [entry[0] for entry in X_scaled]

    # Quantile normalizer (normal distribution)
    # transformer = QuantileTransformer()

    # Quantile normalizer (uniform distribution)
    transformer = QuantileTransformer(
        n_quantiles=1000,
        output_distribution='uniform',
    )
    X_scaled = transformer.fit_transform(X)
    X_scaled = [entry[0] for entry in X_scaled]

    # Normalize by dividing to max
    # X_max = max(data_graph['w'])
    # X_scaled = [entry[0] / X_max for entry in X]

    # Fixing 0's and 1's entries
    # X_scaled = [entry if entry != 1 else 0.99 for entry in X_scaled]
    # X_scaled = [entry if entry > 0 else 0.1 for entry in X_scaled]

    # Scale everything between [a,b] or [0.5,1]
    # Because we do not want these weight become less than temporal weights
    # X_scaled = (b - a) * ((X_scaled - min(X_scaled)) / (max(X_scaled) - min(X_scaled))) + a
    X_scaled = (0.5 * np.array(X_scaled)) + 0.5

    # Rounding to X decimal point
    # X_scaled = [round(entry, 2) for entry in X_scaled]  # List
    X_scaled = np.round(X_scaled, 2)  # Array

    # Plot the distribution of scaled weights
    if plot_weights:
        plt.figure()
        ax = sns.distplot(
            X_scaled,
            bins=max(X),
            kde=True,
            hist_kws={
                "linewidth": 15,
                'alpha': 1
            }
        )
        ax.set(xlabel='Scaled Edge Weight', ylabel='Frequency')

    # Save back scaled weights to DF
    data_graph['w'] = X_scaled

    # Save scaled weights to file
    if save_weights:
        np.savetxt(file_out[10], X_scaled, delimiter=',', fmt='%s')

    # Save network to DB
    if save_network_db:
        data_graph[['u', 'v', 't', 'w']].to_sql(
            name='bluetooth_edgelist',
            con=sqlite3.connect(path2),
            if_exists='replace',
            index_label='id'
        )

    # Save network to file
    if save_network_csv:
        data_graph[['u', 'v', 't',
                    'w']].to_csv(file_out[2], header=False, index=False)

    # Add edges to network object
    for row in data_graph.itertuples(index=True, name='Pandas'):
        graph.add_edge(
            getattr(row, 'u'),
            getattr(row, 'v'),
            t=getattr(row, 't'),
            w=getattr(row, 'w')
        )

    # Save graph to file as netX object
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save timestamps
    if save_times:
        times_set = set()
        for u, v, w in graph.edges(data=True):
            times_set.add(w['t'])
        times = pd.Series(sorted(list(times_set)))
        np.savetxt(file_out[4], times, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the graph
        nodes = pd.Series(sorted(list(graph.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_out[6], nodes, delimiter=',', fmt='%s')

    # Bipartite network edge data
    l1, l2, l3, l4 = [], [], [], []
    for k1, v1 in time_bipartite_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_bi_graph = pd.DataFrame(
        list(zip(l1, l2, l3, l4)), columns=['t', 'u', 'v', 'w']
    )

    # Weights
    X = [[entry] for entry in data_bi_graph['w']]
    if save_weights: np.savetxt(file_out[9], X, delimiter=',', fmt='%s')
    transformer = QuantileTransformer(
        n_quantiles=100,
        output_distribution='uniform',
    )
    X_scaled = transformer.fit_transform(X)
    X_scaled = [entry[0] for entry in X_scaled]
    if save_weights:
        np.savetxt(file_out[11], X_scaled, delimiter=',', fmt='%s')
    data_bi_graph['w'] = X_scaled

    # Save bipartite to DB
    if save_network_db:
        data_bi_graph[['u', 'v', 't', 'w']].to_sql(
            name='bluetooth_bipartite_edgelist',
            con=sqlite3.connect(path2),
            if_exists='replace',
            index_label='id'
        )

    # Save bipartite to file
    if save_network_csv:
        data_bi_graph[['u', 'v', 't',
                       'w']].to_csv(file_out[3], header=False, index=False)

    # Add nodes and edges to bipartite network oject
    # We need to add a prefix "u_" for users & "b_" for BT devices to the node id
    # So that we can distinguish them from each others
    for row in data_bi_graph.itertuples(index=True, name='Pandas'):
        # In bluetooth connections, user devices ID are repeated in all BT devices
        # So there is no need to differentiate between them
        bipartite.add_edge(
            getattr(row, 'u'),
            getattr(row, 'v'),
            t=getattr(row, 't'),
            w=getattr(row, 'w')
        )

    # Save graph
    if save_network_file:
        nx.write_gpickle(bipartite, file_out[1])

    # Save timestamps
    if save_times:
        times_set = set()
        for u, v, w in bipartite.edges(data=True):
            times_set.add(w['t'])
        times_bipartite = pd.Series(sorted(list(times_set)))
        np.savetxt(file_out[5], times_bipartite, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the bipartite graph
        nodes = pd.Series(sorted(list(bipartite.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_out[7], nodes, delimiter=',', fmt='%s')
        # pd.DataFrame(sorted(list(times))).to_csv(file_in[2], header=None, index=False)

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        print(f'Number of times = {len(times)}')

    return graph


def temporal_bt_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Network
    graph = nx.MultiDiGraph()

    # Read from file
    if os.path.exists(file_in[0]):
        if output: print('Reading temporal network ...')
        graph = nx.read_gpickle(file_in[0])
    else:
        if output: print('Temporal network file was not found')
        return None

    # Print graph statistics
    if output: print(nx.info(graph))

    return graph


def temporal_bt_times_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_times.csv'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads timestamps of temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Times
    times = []

    # Read from file
    if os.path.exists(file_in[0]):
        if output: print('Reading times ...')
        times = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['times']
        ).iloc[:, 0]
        # Change type (str -> datetime)
        times = times.astype('datetime64[ns]')
    else:
        if output: print('Times file was not found')
        return None

    return times


def temporal_bt_nodes_read(
    folder_in=NETWORK,
    file_in=['bt_temporal_nodes.csv'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads nodes of temporal graph
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Nodes
    nodes = []

    # Read from file
    if os.path.exists(file_in[0]):
        if output: print('Reading nodes ...')
        nodes = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['nodes']
        ).iloc[:, 0]
    else:
        if output: print('Nodes file was not found')
        return None

    return nodes


# --------------
# Static Network
# --------------


def static_bt(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=['bt_temporal_network.gpickle'],
    file_out=['bt_static_network.gpickle', 'bt_static_edgelist.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    tmies=None,
    undirected=True,
    output_network=False,
    save_network_csv=False,
    save_network_file=False,
):
    """
    Convert input temporal network to the (aggregated) static network
    the edge weights are sum of temporal interactions over entire time window
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal network from file
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in,
            file_in,
            label_folder_in,
            label_file_in,
        )

    # Create static graph
    graph = nx.Graph()
    graph.name = 'Static Network'

    # Update static network edges
    for u, v, data in temporal.edges(data=True):
        t = 1 if 't' in data else 0
        if graph.has_edge(u, v):
            graph[u][v]['s'] += t
            graph[u][v]['w'] = graph[u][v]['w'] + data['w']
        else:
            graph.add_edge(u, v, s=t, w=data['w'])

    # Fix edge weights, if network is directed, because they have been counted twice
    if undirected:
        for u, v, data in graph.edges(data=True):
            graph[u][v]['s'] //= 2
            graph[u][v]['w'] /= 2
            # Mean of weight
            graph[u][v]['w'] /= graph[u][v]['s']

    # Save the network
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    if save_network_csv:
        # (1)
        # nx.write_edgelist(graph, file_out[1], data=True)
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Densitiy
    den = nx.classes.function.density(graph)
    # Is network connected
    con = nx.algorithms.components.is_connected(graph)
    # Connected components
    cc = 1
    # Diameter
    dim = graph.number_of_nodes()
    if not con:
        cc = nx.algorithms.components.number_connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        dim = nx.algorithms.distance_measures.diameter(largest_cc)
    else:
        dim = nx.algorithms.distance_measures.diameter(graph)

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        if con:
            print('Network is connected.')
        else:
            print('Network is not connected.')
        print('Density =', den)
        print('Number of connected components =', cc)
        print('Diameter =', dim)

    return graph


def static_bt_read(
    folder_in=NETWORK,
    file_in=['bt_static_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
    stat=False,
):
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)

    # Static graph
    graph = nx.Graph()

    # Read the network from file
    if os.path.exists(file_in[0]):
        if output: print('Reading static network ...')
        graph = nx.read_gpickle(file_in[0])

    # Network statistics
    if stat:
        # Densitiy
        den = nx.classes.function.density(graph)
        # Is network connected
        con = nx.algorithms.components.is_connected(graph)
        # Connected components
        cc = 1
        # Diameter
        dim = graph.number_of_nodes()
        if not con:
            cc = nx.algorithms.components.number_connected_components(graph)
            largest_cc = max(nx.connected_components(graph), key=len)
            dim = nx.algorithms.distance_measures.diameter(largest_cc)
        else:
            dim = nx.algorithms.distance_measures.diameter(graph)

        # Print network statistics
        if output:
            print(nx.info(graph))
            if con:
                print('Network is connected.')
            else:
                print('Network is not connected.')
            print('Density =', den)
            print('Number of connected components =', cc)
            print('Diameter =', dim)

    return graph


# --------------------
# Time-Ordered Network
# --------------------


def ton_bt(
    folder_in=[DB, NETWORK],
    folder_out=NETWORK,
    file_in=[
        'uiuc.db',
        'bt_temporal_network.gpickle',
        'bt_temporal_times.csv',
    ],
    file_out=['bt_ton_network.gpickle', 'bt_ton_edgelist.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    times=None,
    directed=True,
    teleport=False,
    loop=False,
    trans=True,
    output_network=False,
    save_network_db=False,
    save_network_csv=True,
    save_network_file=True
):
    """
    Create a (directed) time-ordered (temporal) network
    the LIGHT version do not set any edge weight attribute, keeping the model light
    
    Parameters
    ----------
    directed : bool
        add bi-directional temporal edges i.e. t <-> t+1
    teleport :bool
        add temporal teleportation edges
    loop : bool
        connect nodes at last timestamp to first i.e. T -> t0
    """
    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal networks and timestamps
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in[1],
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    if times is None:
        times = temporal_bt_times_read(
            folder_in[1],
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )

    # TON graph
    graph = nx.DiGraph()
    graph.name = 'Time-Ordered Network'

    # Size of timestamp list
    T = len(times)

    # Time -> Index
    time_index = dict((v, i) for i, v in enumerate(times))

    # Size of nodes
    N = temporal.number_of_nodes()

    # Node -> Index
    nodes = pd.Series(sorted(list(temporal.nodes)))
    node_index = dict((v, i) for i, v in enumerate(nodes))

    # Size of edges
    L = temporal.number_of_edges()

    # Node (index) -> horizontal edges
    node_edges = {}
    for n in range(N):
        node_edges[n] = [(N * t + n, N * (t + 1) + n) for t in range(T)]

    # Horizontal edges
    if directed:
        for node, edges in node_edges.items():
            for i in range(len(edges)):  # [0,T]
                # Add edges: node(i) -> node(i+1)
                graph.add_edge(edges[i][0], edges[i][1])
                # With temporal teleportation
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(edges[i][0], edges[j][1])
        # With temporal loop
        if loop:
            for node in node_edges:
                graph.add_edge(node_edges[node][-1][1], node_edges[node][0][0])
    else:  # Undirected
        for node, edges in node_edges.items():
            for i in range(len(edges)):
                graph.add_edge(edges[i][0], edges[i][1])
                # Backward horizontal edge (i.e. moving back in time)
                graph.add_edge(edges[i][1], edges[i][0])
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(edges[i][0], edges[j][1])
                        # Backward teleportation in time
                        graph.add_edge(edges[j][1], edges[i][0])
        # With tempora loop
        if loop:
            for node in node_edges:
                graph.add_edge(node_edges[node][-1][1], node_edges[node][0][0])
                graph.add_edge(node_edges[node][0][0], node_edges[node][-1][1])

    # Crossed edges
    if directed:  # Directed
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                w=edge_data['w']  # Only edge weight is set in light version
            )
    else:  # Undirected
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                w=edge_data['w']
            )
            graph.add_edge(
                v_index + (t_index + 1) * N,
                u_index + t_index * N,
                w=edge_data['w']
            )

    # Transitive closure
    trans_num = 0
    if trans:
        for t in range(T):
            snap_nodes = [(t * N) + n for n in range(N)]
            snap_nodes.extend([((t + 1) * N) + n for n in range(N)])
            snap_graph = graph.subgraph(snap_nodes)
            A = nx.to_numpy_matrix(snap_graph)
            A_t = A[:len(A) // 2, len(A) // 2:]
            snap_trans = nx.to_numpy_matrix(
                nx.transitive_closure(
                    nx.from_numpy_matrix(A_t, create_using=nx.DiGraph)
                )
            )
            # Compare edges of transitive closure with edges we had before
            # Find new edges, add them to network
            snap_edges = np.transpose(np.nonzero(A_t != snap_trans))
            snap_weights = np.tile(
                0.5 * np.random.sample(len(snap_edges) // 2) + 0.5, 2
            )
            # index of new edges should be converted into node ID in network
            for r in range(len(snap_edges)):
                if not graph.has_edge(
                    snap_nodes[snap_edges[r][0]],
                    snap_nodes[snap_edges[r][1] + N]
                ):
                    trans_num += 1  # Counter of transitive edges
                    graph.add_edge(
                        snap_nodes[snap_edges[r][0]],
                        snap_nodes[snap_edges[r][1] + N],
                        w=snap_weights[r],
                        trans=True
                    )
                    if not directed:
                        graph.add_edge(
                            snap_nodes[snap_edges[r][0]] + N,
                            snap_nodes[snap_edges[r][1]],
                            w=snap_weights[r],
                            trans=True
                        )

    # Save network to file
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save network edgelist
    if save_network_csv:
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Save network to database
    if save_network_db:
        edge_list = pd.DataFrame.from_dict(graph.edges)
        edge_list.columns = ['u', 'v']
        edge_list.to_sql(
            name='bluetooth_time_ordered_edgelist',
            con=sqlite3.connect(path1),
            if_exists='replace',
            index_label='id'
        )

    # Print network statics
    if output_network:
        print(nx.info(graph))
        if trans:
            print(f'Number of transitive edges = {trans_num}')

    return graph


def ton_bt_full(
    folder_in=[DB, NETWORK],
    folder_out=NETWORK,
    file_in=[
        'uiuc.db',
        'bt_temporal_network.gpickle',
        'bt_temporal_times.csv',
    ],
    file_out=[
        'bt_tonf_network.gpickle',
        'bt_tonf_edgelist.csv',
        'bt_tonf_delta.csv',
        'bt_tonf_weights.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    times=None,
    directed=True,
    trans=True,
    teleport=False,
    loop=False,
    output_delta=False,
    output_network=False,
    save_delta=False,
    save_network_csv=True,
    save_network_file=True
):
    """
    Create a (directed) time-ordered (temporal) network
    the FULL version set a nubmer of edge weight attributes as position and color
    
    Parameters
    ----------
    directed : bool
        add bi-directional temporal edges i.e. t <-> t+1
    teleport :bool
        add temporal teleportation edges
    loop : bool
        connect nodes at last timestamp to first i.e. T -> t0
    """
    # Edit paths
    path1 = path_edit(
        [file_in[0]],
        folder_in[0],
        label_file_in,
        label_folder_in,
    )[0]
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Read temporal networks and timestamps
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in[1],
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    if times is None:
        times = temporal_bt_times_read(
            folder_in[1],
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )

    # TON graph
    graph = nx.DiGraph()
    graph.name = 'Full Time-Ordered Network'

    # Size of timestamp list
    T = len(times)

    # Time -> Index
    time_index = dict((v, i) for i, v in enumerate(times))

    # Size of nodes
    N = temporal.number_of_nodes()

    # Node -> Index
    nodes = pd.Series(sorted(list(temporal.nodes)))
    node_index = dict((v, i) for i, v in enumerate(nodes))

    # Size of edges
    L = temporal.number_of_edges()

    # Node (index) -> horizontal edges
    node_edges = {}
    for n in range(N):
        node_edges[n] = [(N * t + n, N * (t + 1) + n) for t in range(T)]

    # Colors for nodes at differnt timestamp
    colors = []
    cmap = cm.get_cmap('Wistia', T + 1)
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))

    # Time delta or time difference between all consecutive timestamps
    # First convert delta to second and then hour
    # Add '1' to the begginig so that len(delta) == len(times)
    delta = []
    times = list(times)
    delta_temp = pd.Series(pd.Series(times[1:]) - pd.Series(times[:T - 1]))
    delta = [int(ts.total_seconds() // 3600) for ts in delta_temp]
    delta.insert(0, 1)  # (postion, value)

    # Change time and delta to series
    times = pd.Series(times)
    delta = pd.Series(delta)

    if save_delta:  # save delta
        np.savetxt(file_out[2], delta.values, delimiter=',', fmt='%s')

    if output_delta:
        # delta_count = pd.Series(Counter(delta)).sort_index()
        delta_count = delta.value_counts(normalize=True)
        print("Delta Distribution\n------------------")
        print(delta_count)

    # Horizontal edges
    for node, edges in node_edges.items():
        # Add the first node at the first timestamp
        graph.add_node(node, c=colors[0], p=(0, node))
        for i in range(len(edges)):  # i = time, and in range [0,T]
            # Add the edge (u,v)
            graph.add_edge(
                edges[i][0],
                edges[i][1],
                t=times[i],  # Timestamp
                d=delta[i],  # Delta or temporal distance
                c='silver'  # Color
            )
            # Backward horizontal edge (i.e. moving back in time)
            if not directed:
                graph.add_edge(
                    edges[i][1],
                    edges[i][0],
                    t=times[i],
                    d=delta[i],
                    c='silver'
                )
            # Then set attribute of second node of just created edge (u,v)
            graph.nodes[edges[i][1]]['c'] = colors[i + 1]
            graph.nodes[edges[i][1]]['p'] = (i + 1, node)
            # Temporal teleportation
            if teleport:
                for j in range(i + 1, len(edges)):
                    graph.add_edge(
                        edges[i][0], edges[j][1], d=sum(delta[i:j]), c='gold'
                    )
                    if not directed:
                        graph.add_edge(
                            edges[j][0],
                            edges[i][1],
                            d=sum(delta[i:j]),
                            c='gold'
                        )
    # Temporal loop
    if loop:
        for node in node_edges:
            graph.add_edge(
                node_edges[node][-1][1],
                node_edges[node][0][0],
                d=sum(delta),
                c='orange'
            )
            if not directed:
                graph.add_edge(
                    node_edges[node][0][0],
                    node_edges[node][-1][1],
                    d=sum(delta),
                    c='orange'
                )

    # Crossed edges
    for u, v, edge_data in temporal.edges(data=True):
        u_index = node_index[u]
        v_index = node_index[v]
        t_index = time_index[edge_data['t']]
        graph.add_edge(
            u_index + t_index * N,
            v_index + (t_index + 1) * N,
            w=edge_data['w'],
            t=edge_data['t'],
            d=delta[t_index],
            c='black'
        )
        if not directed:
            graph.add_edge(
                v_index + (t_index + 1) * N,
                u_index + t_index * N,
                w=edge_data['w'],
                t=edge_data['t'],
                d=delta[t_index],
                c='black'
            )

    # Transitive closure
    trans_num = 0
    if trans:
        for t in range(T):
            snap_nodes = [(t * N) + n for n in range(N)]
            snap_nodes.extend([((t + 1) * N) + n for n in range(N)])
            snap_graph = graph.subgraph(snap_nodes)
            A = nx.to_numpy_matrix(snap_graph)
            A_t = A[:len(A) // 2, len(A) // 2:]
            snap_trans = nx.to_numpy_matrix(
                nx.transitive_closure(
                    nx.from_numpy_matrix(A_t, create_using=nx.DiGraph)
                )
            )
            # Compare edges of transitive closure with edges we had before
            # Find new edges, add them to network
            snap_edges = np.transpose(np.nonzero(A_t != snap_trans))
            snap_weights = np.tile(
                0.5 * np.random.sample(len(snap_edges) // 2) + 0.5, 2
            )
            # index of new edges should be converted into node ID in network
            for r in range(len(snap_edges)):
                if not graph.has_edge(
                    snap_nodes[snap_edges[r][0]],
                    snap_nodes[snap_edges[r][1] + N]
                ):
                    trans_num += 1  # Counter of transitive edges
                    graph.add_edge(
                        snap_nodes[snap_edges[r][0]],
                        snap_nodes[snap_edges[r][1] + N],
                        w=snap_weights[r],
                        t=times[t],
                        d=delta[t],
                        trans=True,
                        c='red'
                    )  # Only for trans edges
                    if not directed:
                        graph.add_edge(
                            snap_nodes[snap_edges[r][0]] + N,
                            snap_nodes[snap_edges[r][1]],
                            w=snap_weights[r],
                            t=times[t],
                            d=delta[t],
                            trans=True,
                            c='red'
                        )

    # Save network to file
    if save_network_file:
        nx.write_gpickle(graph, file_out[0])

    # Save network edgelist
    if save_network_csv:
        nx.write_weighted_edgelist(graph, file_out[1], delimiter=',')

    # Print network statics
    if output_network:
        print(nx.info(graph))
        if trans:
            print(f'Number of transitive edges = {trans_num}')

    return graph


def ton_bt_to_temporal(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    file_out=[
        'bt_temporal_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    temporal=None,
    trans_remove=True,
    output_network=True,
    save_times=True,
    save_nodes=True,
    save_network_file=True
):
    """
    Convert (directed) time-ordered network (TON) to temporal network
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Create empty network
    graph = nx.MultiDiGraph()
    graph_name = 'Temoral Network'
    if len(label_file_out) > 0: graph_name = graph_name + ' ' + label_file_out
    graph.name = graph_name

    # Read temporal networks from file
    if temporal is None:
        temporal = temporal_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Nodes
    nodes = temporal_bt_nodes_read(
        folder_in,
        [file_in[1]],
        label_folder_in,
        label_file_in,
    )
    N = file_line_count(
        path_edit(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )[0]
    )
    # N = len(nodes)

    # Timestamp
    times = temporal_bt_times_read(
        folder_in,
        [file_in[2]],
        label_folder_in,
        label_file_in,
    )
    T = file_line_count(
        path_edit(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )[0]
    )
    # T = len(times)

    # Calculate current timestamp list from graph
    # In case, in node removal caused loosing some timestamp comparing to original
    times_set = set()

    # Iterate edges and add crossed ones back to temporal network object
    for u, v, data in temporal.edges(data=True):
        parent_u = u % N
        parent_v = v % N
        time_uv = u // N  # OR v // N - 1
        time_delta = abs(v - u) // N
        # Crossed edge
        if parent_u != parent_v:  # and time_delta == 1:
            if trans_remove and data.get('trans', False):
                # If the the edge is transitive and we want to ignore trans -> skip
                continue
            graph.add_edge(
                parent_u,
                parent_v,
                t=times.loc[time_uv],
                w=data['w'],
            )
            # Save timestamp to the new time set
            times_set.add(times.loc[time_uv])

    # Convert times set to series and save
    times_new = pd.Series(sorted(list(times_set)))
    nodes_new = pd.Series(sorted(list(graph.nodes)))

    # Save graph
    if save_network_file: nx.write_gpickle(graph, file_out[0])

    # Save nodes
    if save_nodes:
        np.savetxt(file_out[1], nodes_new, delimiter=',', fmt='%s')

    # Save times
    if save_times:
        np.savetxt(file_out[2], times_new, delimiter=',', fmt='%s')

    # Print network statistics
    if output_network:
        print(nx.info(graph))
        print(f'Number of times: {len(times_new)}')

    return graph


def ton_bt_read(
    folder_in=NETWORK,
    file_in=['bt_ton_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    Reads time-ordered network (TON) graph of Bluetooth connections
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)
    graph = nx.read_gpickle(file_in[0])
    if output: print(nx.info(graph))
    return graph


def ton_bt_full_read(
    folder_in=NETWORK,
    file_in=['bt_ton_full_network.gpickle'],
    label_folder_in='',
    label_file_in='',
    output=False,
):
    """
    bt_ton_full_network.gpickle
    Reads full version of time-ordered network (TON) graph of Bluetooth connections
    """
    # Edit paths
    file_in = path_edit(file_in, folder_in, label_file_in, label_folder_in)
    graph = nx.read_gpickle(file_in[0])
    if output: print(nx.info(graph))
    return graph


def ton_bt_analyze(
    folder_in=NETWORK,
    file_in=[
        'bt_temporal_network.gpickle',
        'bt_temporal_times',
        'bt_ton_network.gpickle',
    ],
    label_folder_in='',
    label_file_in='',
    output=False,
    plot=True,
):
    """
    Calculate sum of outdegree of nodes over time & node
    and tries to fit it to powerlaw and lognormal distributions
    """
    temporal = temporal_bt_read(
        folder_in,
        [file_in[0]],
        label_folder_in,
        label_file_in,
    )
    times = temporal_bt_times_read(
        folder_in,
        [file_in[1]],
        label_folder_in,
        label_file_in,
    )
    graph = ton_bt_read(
        folder_in,
        [file_in[2]],
        label_folder_in,
        label_file_in,
    )

    # Size of nodes, edges and times
    N = temporal.number_of_nodes()
    L = temporal.number_of_edges()
    T = len(times)
    # N_new = graph.number_of_nodes()
    # L_new = graph.number_of_edges()

    # Dictionary {time -> id of nodes in that time}
    time_nodes = {}
    for t in range(T):
        time_nodes[t] = [N * t + n for n in range(N)]

    # Check the edge frequency in each timestamp
    time_out_degrees = {}
    for t in sorted(time_nodes):  # t in [0 ... T]
        time_out_degrees[t] = [graph.out_degree(n) - 1 for n in time_nodes[t]]

    # Dataframe of outdegress with time as columns
    time_out_degrees = pd.DataFrame.from_dict(time_out_degrees)

    if output:
        print(
            sorted(
                Counter(time_out_degrees.sum(0)).items(), key=lambda x: x[0]
            )
        )

    # Powerlaw correlation of sum of outdegree of nodes over time
    out_degrees_sum = time_out_degrees.sum(0)
    pl = powerlaw.Fit(out_degrees_sum)
    R, p = pl.distribution_compare('power_law', 'lognormal')
    if plot:
        print(pl.power_law.alpha)
        print(pl.power_law.xmin)
        print(R, p)
    if output:
        # Max normalize sum of out degrees [0,1]
        print(out_degrees_sum / max(out_degrees_sum))
        # Sum of out degrees over nodes
        print(time_out_degrees.sum(1))


# ------------
# Edge Weights
# ------------


def ew(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    file_out=['bt_ton_weights.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    nodes=None,
    times=None,
    directed=True,
    teleport=False,
    loop=False,
    version=0,
    omega=1,
    epsilon=1,
    gamma=0.0001,
    distance=1,
    alpha=0.5,
    save_weights=False,
    output_weights=False,
    plot_weights=False
):
    """
    Calculate (dynamic) weights for (horizontal) temporal edges of TON model
    
    Parameters
    ----------
    graph : NetworkX
        time-ordered network (TON)
    number_of_nodes : int
        numbero of nodes from the temporal graph (default is None, then reads from file)
    number_of_times : int
        numbero of timestamps from the temporal graph (default is None, then reads from file)
    version : int
        0 -> contact value of omega
        1 -> dynamic (alpha)^(lengh_of_horizontal_path)
        2 -> dynamic (1)/(lenght_of_horizontal_path)
        3 -> dynamic (1)/(log2(lenght_of_horizontal_path))
    omega : float
        weight factor of horizontal edges (e.g. 1, 0.01, 0.001, 0.0001 ...)
    epsilon :float
        weight factor of crossed edgess (e.g. 1, 0.01, 0.001, 0.0001 ...)
    gamma : float
        weight of horizontal teleport edges (e.g. 0.0001, 0.001, 0.01 ...)
    distance : float
        value that being added to as lenght to non-active consecutive edges or paths (smaller -> slower weight decay)
    alpha : float
        magnification factor in version 1 (larger -> slower weight decay), default = 1/2 or 0.5
    
    Returns
    -------
    dict
        {(u,v):temporal_weight}
    """
    def has_crossed_edge(in_graph, in_N, in_node):
        """
        Detect if input node in TON graph has incoming crossed edges
        or it only has incoming horizontal edges from itself in past timestamp
        """
        in_parent = in_node % in_N
        for pre in in_graph.predecessors(in_node):
            if pre % in_N != in_parent:
                return True
        # Else = no crossed edge found ...
        return False

    def ton_features(graph):
        """
        Detect if TON graph is (1) directed, has (2) teleportation (3) temporal loop
        """
        # TODO: so far, only works if TON is not altered (no node has been removed)
        directed, teleport, loop = True, False, False
        # If node 0 at time 1 connects to node 0 at time 0 graph is undirected
        if (1 * N) + 0 in graph.predecessors(0):
            directed = False

        # If node 0 at time 0 connects to node 0 at time 2 graph has teleport edges
        if (2 * N) + 0 in graph.successors(0):
            teleport = True

        # If node 0 at time T connects to node 0 at time 0 graph has temporal loop
        if (T * N) + 0 in graph.predecessors(0):
            loop = True

    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # TON graph
    if graph is None:
        graph = ton_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
        N = file_line_count(
            path_edit(
                folder_in,
                [file_in[1]],
                label_folder_in,
                label_file_in,
            )[0]
        )
    else:
        N = len(nodes)

    # Timestamp
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
        N = file_line_count(
            path_edit(
                folder_in,
                [file_in[2]],
                label_folder_in,
                label_file_in,
            )[0]
        )
    else:
        T = len(times)

    # Edge-Weight dictionary {(u,v):weight}
    ew = {}

    # Horizontal edges as helper dictionary
    hedges = {}

    # If node in_degree = 1 then node 'u' has only '1' horizontal-edge of (v,u)
    nid = 1

    if version == 0:  # Without penalization
        for u, v in graph.edges():
            # When u & v have same parent -> edge (u,v) is horizontal
            time_delta = abs(v - u) // N
            parent_u = u % N
            parent_v = v % N
            if parent_u != parent_v:  # Crossed
                ew[(u, v)] = epsilon  # E.g. 1
            else:
                if time_delta > 1:  # Teleport
                    ew[(u, v)] = gamma  # E.g. 0.0001
                else:  # # Horizontal OR time_delta = 1
                    # Node v is node u at one timestamp after
                    ew[(u, v)] = omega

    else:  # Penalize ...
        # Nodes [0-N]
        for u, v in sorted(graph.edges(), key=lambda x: x[0]):
            time_delta = abs(v - u) // N
            parent_u = u % N
            parent_v = v % N
            if parent_u != parent_v:
                ew[(u, v)] = epsilon
            else:
                if time_delta > 1:
                    ew[(u, v)] = gamma
                else:
                    # Node v has crossed edge
                    # if graph.in_degree(v) != nid:  # 1 or 2
                    if has_crossed_edge(graph, N, v):
                        hedges[(u, v)] = omega  # E.g. 1
                    else:
                        # Node v does not have crossed edge
                        # Look at the previous edge weight (if exsit, otherwise return omega)
                        hedges[(u, v)
                               ] = hedges.get((u - N, u), omega) + distance

    # Update weights based on version of penalization
    if version == 1:
        # Decay exponentially fast
        # (parameteralpha)^(distance) e.g. 1/2^1 , 1/2^2, ...
        for edge, weight in hedges.items():
            hedges[edge] = alpha**(weight - 1)
    elif version == 2:
        # Decay very fast
        # 1/(distance) e.g. 1/2, 1/3, ...
        for edge, weight in hedges.items():
            hedges[edge] = 1 / weight
    elif version == 3:
        # Decay fast
        # 1/log2(distance + 1) e.g. 1/log2, 1/log3, ...
        for edge, weight in hedges.items():
            hedges[edge] = 1 / np.log2(weight + 1)

        # Finish by updating helper dictionary to 'ew'
        ew.update(hedges)

    if save_weights:
        pd.Series(ew).reset_index().to_csv(
            file_out[0],
            header=False,
            index=False,
        )

    if output_weights:
        for e, w in sorted(ew.items(), key=lambda x: x[0]):
            if e[0] % N == e[1] % N:  # H
                if graph.in_degree(e[1]) == nid:
                    print('{}\t->\t{}\t{}'.format(e[0], e[1], w))

    if plot_weights and version > 0:
        ls = sorted(ew.items())
        ls1, ls2 = zip(*ls)
        plt.figure()
        ax = sns.histplot(
            ls2,
            bins=max(ls2),
            kind='kde',
            hist_kws={
                "linewidth": 15,
                'alpha': 1
            }
        )
        ax.set(xlabel='Horizontal Edge Weight', ylabel='Frequency')

    return ew


# -----------------------------
# Transivity Probability Matrix
# -----------------------------


def prob(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
    ],
    file_out=['bt_ton_weights.csv', 'bt_ton_probs.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    nodes=None,
    times=None,
    directed=True,
    teleport=False,
    loop=False,
    version=0,
    omega=1,
    epsilon=1,
    gamma=0.0001,
    distance=1,
    alpha=0.5,
    save_weights=True,
    save_probs=True,
    output_probs=False,
    plot_probs=False,
):
    """
    In addition to horizontal edge weights ...
    this method, create edge transmission probability applicable in spread
    """
    def has_crossed_edge(in_graph, in_N, in_node):
        """
        Detect if input node in TON graph has incoming crossed edges
        or it only has incoming horizontal edges from itself in past timestamp
        """
        in_parent = in_node % in_N
        for pre in in_graph.predecessors(in_node):
            if pre % in_N != in_parent:
                return True
        # Else = no crossed edge found ...
        return False

    def ton_features(graph):
        """
        Detect if TON graph is (1) directed, has (2) teleportation (3) temporal loop
        """
        directed, teleport, loop = True, False, False
        # If node 0 at time 1 connects to node 0 at time 0 graph is undirected
        if (1 * N) + 0 in graph.predecessors(0):
            directed = False

        # If node 0 at time 0 connects to node 0 at time 2 graph has teleport edges
        if (2 * N) + 0 in graph.successors(0):
            teleport = True

        # If node 0 at time T connects to node 0 at time 0 graph has temporal loop
        if (T * N) + 0 in graph.predecessors(0):
            loop = True

    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # TON graph
    if graph is None:
        graph = ton_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
        N = file_line_count(
            path_edit(
                folder_in,
                [file_in[1]],
                label_folder_in,
                label_file_in,
            )[0]
        )
    else:
        N = len(nodes)

    # Timestamp
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
        N = file_line_count(
            path_edit(
                folder_in,
                [file_in[2]],
                label_folder_in,
                label_file_in,
            )[0]
        )
    else:
        T = len(times)

    # Edge-Weight dictionary {(u,v):weight}
    ew = {}

    # Horizontal edges as helper dictionary
    hedges = {}

    # If node in_degree = 1 then node 'u' has only '1' horizontal-edge of (v,u)
    nid = 1

    if version == 0:  # Without penalization
        for u, v in graph.edges():
            # When u & v have same parent -> edge (u,v) is horizontal
            time_delta = abs(v - u) // N
            parent_u = u % N
            parent_v = v % N
            if parent_u != parent_v:  # Crossed
                ew[(u, v)] = epsilon  # E.g. 1
            else:
                if time_delta > 1:  # Teleport
                    ew[(u, v)] = gamma  # E.g. 0.0001
                else:  # # Horizontal OR time_delta = 1
                    # Node v is node u at one timestamp after
                    ew[(u, v)] = omega

    else:  # Penalize ...
        # Nodes [0-N]
        for u, v in sorted(graph.edges(), key=lambda x: x[0]):
            time_delta = abs(v - u) // N
            parent_u = u % N
            parent_v = v % N
            if parent_u != parent_v:
                ew[(u, v)] = epsilon
            else:
                if time_delta > 1:
                    ew[(u, v)] = gamma
                else:
                    # Node v has crossed edge
                    # if graph.in_degree(v) != nid:  # 1 or 2
                    if has_crossed_edge(graph, N, v):
                        hedges[(u, v)] = omega  # E.g. 1
                    else:
                        # Node v does not have crossed edge
                        # Look at the previous edge weight (if exsit, otherwise return omega)
                        hedges[(u, v)
                               ] = hedges.get((u - N, u), omega) + distance

    # Update weights based on version of penalization
    if version == 1:
        # Decay exponentially fast
        # (parameteralpha)^(distance) e.g. 1/2^1 , 1/2^2, ...
        for edge, weight in hedges.items():
            hedges[edge] = alpha**(weight - 1)
    elif version == 2:
        # Decay very fast
        # 1/(distance) e.g. 1/2, 1/3, ...
        for edge, weight in hedges.items():
            hedges[edge] = 1 / weight
    elif version == 3:
        # Decay fast
        # 1/log2(distance + 1) e.g. 1/log2, 1/log3, ...
        for edge, weight in hedges.items():
            hedges[edge] = 1 / np.log2(weight + 1)

        # Finish by updating helper dictionary to 'ew'
        ew.update(hedges)

    if save_weights:
        pd.Series(ew).reset_index().to_csv(
            file_out[0],
            header=False,
            index=False,
        )

    # Edge-Probability dictionary {(u,v):p}
    prob = ew.copy()
    
    # Initialize from edge weights i.e. prob has weight of horizontal edges
    # Optionally, we can scale down weight of horizontal to range [0-0.5]
    # Next we read the rest of edge weights e.g. crossed, teleport and loop
    # Add all of them to prob dictionary and then normalize over nodes out-degree

    # TODO
    for u, v, data in graph.edges(data=True):
        # u & v have same parent => (u,v) is horizontal
        time_delta = abs(v - u) // N
        parent_u = u % N
        parent_v = v % N
        if parent_u != parent_v:  # Crossed
            ew[(u, v)] = epsilon  # E.g. 1
        else:  # Horizontal
            if time_delta > 1:  # Teleport
                # If _teleport = False during TN, these edges do not exist
                ew[(u, v)] = gamma  # E.g. 0.0001
            else:  # time_delta = 1
                # Node v is node u at one timestamp after
                ew[(u, v)] = omega

    # X_scaled = (b - a) * ((X_scaled - min(X_scaled)) / (max(X_scaled) - min(X_scaled))) + a
    # X_scaled = (0.5 * np.array(X_scaled)) + 0.5

    for n in graph:
        parent_n = n % N
        w_c = []  # Weights of crossed edges
        w_h = []  # Weights of horizontal edges
        for s in graph.successors(n):
            parent_s = s % N
            if parent_n != parent_s:  # Crossed
                # We read the weight of edge coming from aggregated network
                w_c.append(graph[n][s]['w'])
            else:  # Horizontal or teleport ...
                w_h.append(prob[(n, s)])
        # If more than just one horizontal or crossed
        if len(w_c) + len(w_h) > 1:
            w_c_m = 0
            if len(w_c) > 0:
                w_c_m = max(w_c)
            w_h_m = 0
            if len(w_h) > 0:
                w_h_m = max(w_h)
            # Adjust weights of horizontal according to maximum of crossed
            if w_h_m > w_c_m:
                w_h = [item if item < w_c_m else w_c_m for item in w_h]
            # Update probabilities
            for s in graph.successors(n):
                parent_s = s % N
                if parent_n != parent_s:
                    # Option 1
                    # Leave them to original value from temporal network e.g. [0.5,1]
                    # Most likely many are 0.5 which have 50/50 chance of transmission
                    prob[(n, s)] = graph[n][s]['w']
                    # Option 2
                    # Scale up so maximum is 1.0 while rest are relatively scaled up too
                    # prob[(n, s)] = graph[n][s]['w'] / w_c_m
                else:
                    if prob[(n, s)] < w_c_m:
                        # Option 1
                        # Leave it as it is
                        # prob[(n, s)] = prob[(n, s)]
                        pass
                        # Option 2
                        # Normalized it with maximum value
                        # prob[(n, s)] /= w_c_m
                    else:
                        # Means horizontal weight is larger than max(crossed)
                        # I.e. If exist a C(rossed) < 1 => exist H(orizontal) = 1
                        # Option 1
                        # Scale down 'H' to max('C') and end up being normalized to 1.0
                        # prob[(n, s)] = 1.0
                        # Option 2
                        # Scale down to a random value [0.5,1]
                        prob[(n, s)] = 0.5 * np.random.sample() + 0.5

    if save_probs:
        pd.DataFrame.from_dict(prob, orient='index').to_csv(file_output[1])

    # Check the distribution of probabilities
    if output_probs:
        ls = sorted(prob.items())
        ls1, ls2 = zip(*ls)
        plt.figure()
        ax = sns.distplot(ls2, kde=True)
        # ax = sns.distplot([i for i in ls2 if i != 1], kde=True)

    if output_weights:
        for e, w in sorted(ew.items(), key=lambda x: x[0]):
            if e[0] % N == e[1] % N:  # H
                if graph.in_degree(e[1]) == nid:
                    print('{}\t->\t{}\t{}'.format(e[0], e[1], w))

    if plot_weights and version > 0:
        ls = sorted(ew.items())
        ls1, ls2 = zip(*ls)
        plt.figure()
        ax = sns.histplot(
            ls2,
            bins=max(ls2),
            kind='kde',
            hist_kws={
                "linewidth": 15,
                'alpha': 1
            }
        )
        ax.set(xlabel='Edge Probability', ylabel='Frequency')

    return ew, prob