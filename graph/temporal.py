# ------
# Import
# ------

import networkx as nx

from collections import Counter
from collections import defaultdict
from itertools import permutations

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

# TODO
def tn_bt_full_create(
    folder_in=[DB, NETWORK],
    folder_out=NETWORK,
    file_in=[
        'uiuc.db',
        'bt_temporal_network.gpickle',
        'bt_temporal_times.csv',
        'bt_temporal_nodes.csv',
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
    label_input='',
    label_output='',
    directed=True,
    trans=True,
    teleport=False,
    loop=False,
    _color=True,
    _delta=True,
    _delta_type='h',
    version=0,
    _omega=1,
    _gamma=0.0001,
    _epsilon=1,
    _distance=1,
    _alpha=0.5,
    output_delta=False,
    output_weight=False,
    output_network=False,
    save_delta=False,
    save_weight=False,
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

    
    Convert a temporal (aggregated) network to a (directed) time-series (DS) temporal network
    Full version means nodes and edges get labelled with differnt information such as:
        * color
        * position
        * delta time: the time differnec of an edge from previous timestamp
        * temporal weight considering horizontal and crossed format with following versions:
            0: None
            1: (1/2)^(lengh of horizontal path)
            2: 1/(lenght of horizontal path)
            3: 1/(log2(lenght of horizontal path))
    Other parameters are:
        directed: add bi-directional temporal edges i.e. t <-> t+1
        teleport: activate temporal teleportation
        loop: add last timestamp nodes to first i.e. T -> t1
        _color: add color to nodes
        _delta: add delta-time weight to edges
        _delta_type: h (for hour) & m (for minute)
        _omega: horizontal edge weight (by default = 1, without penalize)
        _gamma: teleportation edge weight
        _epsilon: crossed edge weight
        _distance: horizontal edge distance
        _alpha: dynamic penalizing coefficient
        _sigma: crossed teleportation (not used here - applied during HITS)
    """
    # Amend input / output file names using label
    file_input = label_amend(file_input, label_input)
    file_output = label_amend(file_output, label_output)

    # Read temporal networks and its node list from file
    if temporal is None or times is None:
        temporal, times = temporal_bt_read(
            file_input=[file_input[1], file_input[2]]
        )

    # GRAPH
    graph = nx.DiGraph()

    # TIME
    # Size of timestamp list
    T = len(times)
    # Time -> Index
    time_index = dict((v, i) for i, v in enumerate(times))

    # NODE
    # Size of node list
    N = temporal.number_of_nodes()
    # Node -> Index
    nodes = pd.Series(sorted(list(temporal.nodes)))
    node_index = dict((v, i) for i, v in enumerate(nodes))

    # EDGE
    # Node (index) -> horizontal edges
    L = temporal.number_of_edges()
    node_edges = {}
    for n in range(N):
        node_edges[n] = [(N * t + n, N * (t + 1) + n) for t in range(T)]

    # Colors for nodes at differnt timestamp
    colors = []
    if _color:
        cmap = cm.get_cmap('Wistia', T + 1)
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            colors.append(matplotlib.colors.rgb2hex(rgb))
    else:
        colors = ['#000000'] * (T + 1)

    # Create the time difference between all consecutive timestamps
    # Convert delta to second and then hour
    # Add '1' to the begginig so that -> len(delta) == len(times)
    # Because there is nothing before the first timestamp
    # So we assume delta for nodes at the first timestamp is '1' hour
    delta = []
    if _delta:
        times = list(times)
        delta_temp = pd.Series(pd.Series(times[1:]) - pd.Series(times[:T - 1]))
        if _delta_type == 'h':  # Hour scale
            delta = [int(ts.total_seconds() // 3600) for ts in delta_temp]
        elif _delta_type == 'm':  # Minute scale
            delta = [int(ts.total_seconds() // 60) for ts in delta_temp]
        delta.insert(0, 1)  # (postion, value)
        times = pd.Series(times)
    else:
        delta = [1] * T  # 1 hour (default value for all edges)
        times = pd.Series(times)

    if output_delta:
        # Count the unique delta values
        print("Delta time distribution:")
        if _delta_type == 'h':
            delta_count = pd.Series(Counter(delta)).sort_index()
            # print(delta_count[:24])  # Print deltas up to 1 day = 24 hours
            print(delta_count)
        elif _delta_type == 'm':
            # 1 day = 24 H * 60 Min = 1440, anything more than that could be outlier
            # Or some sort of time jumps in the dataset
            delta_count = pd.Series([d for d in delta if d <= 24 * 60]
                                    ).value_counts(normalize=True)
            print(delta_count)

    if save_delta:  # save delta
        np.savetxt(file_output[2], delta, delimiter=',', fmt='%s')
        # pd.DataFrame(delta).to_csv(file_output[2], header=None, index=False)

    # Convert delta list to series for easy access in future
    delta = pd.Series(delta)

    # Horizontal edges
    if directed:
        for node, edges in node_edges.items():
            # Add the first node at the first timestamp
            graph.add_node(
                node,
                # It turned out 'parent' is not necessary, because we use 'node' or parent_index in 'pos'
                # So it can always be extracted from postion of node or id/number_of_nodes
                # parent=nodes[node],  # Parent
                c=colors[0],  # Color
                p=(0, node)
            )  # Position / Cordinates
            for i in range(len(edges)):  # i = time, and in range [0,T]
                # Add the edge (u,v) with its attribute
                graph.add_edge(
                    edges[i][0],
                    edges[i][1],
                    t=times[i],  # Time
                    d=delta[i],  # Delta or temporal distance
                    tw=_omega,  # Temporal weight (tw)
                    c='silver'
                )  # Color

                # Then set node attribute of second node from created edge of (u,v)
                # graph.nodes[edges[i][1]]['parent'] = nodes[node]
                graph.nodes[edges[i][1]]['c'] = colors[i + 1]
                graph.nodes[edges[i][1]]['p'] = (i + 1, node)
                # With temporal teleportation
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(
                            edges[i][0],
                            edges[j][1],
                            # d=sum(delta[i:j])  # Needs testing
                            tw=_gamma,
                            c='gold'
                        )
        # With temporal loop
        if loop:
            for node in node_edges:
                graph.add_edge(
                    node_edges[node][-1][1],
                    node_edges[node][0][0],
                    # d=sum(delta)  # Needs testing
                    tw=_omega,
                    c='orange'
                )
    else:  # Undirected
        for node, edges in node_edges.items():
            graph.add_node(node, c=colors[0], p=(0, node))
            for i in range(len(edges)):
                graph.add_edge(
                    edges[i][0],
                    edges[i][1],
                    t=times[i],
                    d=delta[i],
                    tw=_omega,
                    c='silver'
                )
                # Backward horizontal edge (i.e. moving back in time)
                graph.add_edge(
                    edges[i][1],
                    edges[i][0],
                    t=times[i],
                    d=delta[i],
                    tw=_omega,
                    c='silver'
                )
                graph.nodes[edges[i][1]]['c'] = colors[i + 1]
                graph.nodes[edges[i][1]]['p'] = (i + 1, node)
                if teleport:
                    for j in range(i + 1, len(edges)):
                        graph.add_edge(
                            edges[i][0], edges[j][1], tw=_gamma, c='gold'
                        )
                        # Backward teleportation in time
                        graph.add_edge(
                            edges[j][1], edges[i][0], tw=_gamma, c='gold'
                        )
        # With temporal loop
        if loop:
            for node in node_edges:
                graph.add_edge(
                    node_edges[node][-1][1],
                    node_edges[node][0][0],
                    tw=_omega,
                    c='orange'
                )
                graph.add_edge(
                    node_edges[node][0][0],
                    node_edges[node][-1][1],
                    tw=_omega,
                    c='orange'
                )

    # Crossed edges
    if directed:
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                t=edge_data['t'],
                w=edge_data['w'],
                d=delta[t_index],
                tw=_epsilon,
                c='black'
            )
    else:  # Undirected
        for u, v, edge_data in temporal.edges(data=True):
            u_index = node_index[u]
            v_index = node_index[v]
            t_index = time_index[edge_data['t']]
            graph.add_edge(
                u_index + t_index * N,
                v_index + (t_index + 1) * N,
                t=edge_data['t'],
                w=edge_data['w'],
                d=delta[t_index],
                tw=_epsilon,
                c='black'
            )
            graph.add_edge(
                v_index + (t_index + 1) * N,
                u_index + t_index * N,
                t=edge_data['t'],
                w=edge_data['w'],
                d=delta[t_index],
                tw=_epsilon,
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
            # Compare edges of transitive closure with edges we had before, find new edges, add them to network
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
                        t=times[t],
                        w=snap_weights[r],
                        d=delta[t],
                        tw=_epsilon,
                        trans=True,
                        c='red'
                    )  # Only for trans edges
                    if not directed:
                        graph.add_edge(
                            snap_nodes[snap_edges[r][0]] + N,
                            snap_nodes[snap_edges[r][1]],
                            t=times[t],
                            w=snap_weights[r],
                            d=delta[t],
                            tw=_epsilon,
                            trans=True,
                            c='red'
                        )

    # Prepare for penalizing horizontal edge weights
    # If version == 0 -> skip penalizing all together
    if version != 0:
        for node, edges in node_edges.items():
            for i in range(len(edges)):  # len = T
                # If in-degree second node 'v' of underlying edge (u,v) is 1
                # Meaning it (u,v) is a horizontal edge with no other node connecting to it
                if graph.in_degree(edges[i][1]) == 1:  # 'v' of (u,v)
                    if i == 0:  # There is no edge before so ...
                        graph[edges[i][0]][edges[i][1]][
                            'tw'] = 1 + _distance  # e.g. _distance = 1
                    else:
                        graph[edges[i][0]][edges[i][1]]['tw'] = graph[edges[
                            i - 1][0]][edges[i - 1][1]]['tw'] + _distance

    # (alpha)^(distance) e.g. (1/2)^(1), (1/2)^(2), ...
    # This version is exponential penalization = very harsh
    if version == 1:
        for node, edges in node_edges.items():
            for i in range(len(edges)):
                graph[edges[i][0]][edges[i][1]]['tw'] = _alpha**(
                    graph[edges[i][0]][edges[i][1]]['tw'] - 1
                )

    # 1/(distance)
    # This version is polinomial penalization = harsh :/
    elif version == 2:
        for node, edges in node_edges.items():
            for i in range(len(edges)):
                graph[edges[i][0]][edges[i][1]]['tw'] = 1 / graph[edges[i][0]][
                    edges[i][1]]['tw']

    # 1/log2(distance + 1)
    # This version is logarithmic penalization = not so harsh :)
    elif version == 3:
        for node, edges in node_edges.items():
            for i in range(len(edges)):
                graph[edges[i][0]][edges[i][1]]['tw'] = 1 / np.log2(
                    graph[edges[i][0]][edges[i][1]]['tw'] + 1
                )

    if save_weight:
        ew = {}
        for u, v, tw in graph.edges(data='tw'):
            ew[(u, v)] = tw
        pd.DataFrame.from_dict(ew, orient='index').to_csv(file_output[3])

    # Print temporal weights for horizontal edges
    if output_weight:
        print('node @ time -> edge(u,v) = weight')
        for node, edges in node_edges.items():
            for edge in edges:
                print(
                    '{} @ {}: ({},{}) = {}'.format(
                        node, edge[1] // N, edge[0], edge[1],
                        graph[edge[0]][edge[1]]['tw']
                    )
                )

    if output_network:
        print('Time-ordered Network (Full):')
        print('N =', graph.number_of_nodes())
        print('L =', graph.number_of_edges())
        if _trans:
            print('{} transitive closure edges were added.'.format(trans_num))

    if save_network_file:
        nx.write_gpickle(graph, file_output[0])

    if save_network_csv:
        w_keys = list(
            list(graph.edges(list(graph.nodes(0))[0][0],
                             data=True))[0][2].keys()
        )
        # list of list containing columns of [[node-1],[node-2],[weight-1(e.g. time)],[weight-2],...]
        edgelist_columns = []
        edgelist_columns.append([])
        edgelist_columns.append([])
        for w_id in range(len(w_keys)):
            edgelist_columns.append([])
        for u, v, w in graph.edges(data=True):
            edgelist_columns[0].append(u)
            edgelist_columns[1].append(v)
            for i in range(len(w_keys)):
                edgelist_columns[2 + i].append(w[w_keys[i]])
        pd.DataFrame(edgelist_columns).transpose().to_csv(
            file_output[1], header=False, index=False
        )

    return graph