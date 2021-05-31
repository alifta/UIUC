# ------
# Import
# ------

import networkx as nx

from collections import Counter
from collections import defaultdict
from itertools import permutations
from networkx.algorithms import distance_measures

import scipy.stats as stats
import numpy.linalg as linalg

import math
import timeit
import powerlaw

import matplotlib as mpl
import seaborn as sns
sns.set_style('ticks')

from sklearn.preprocessing import normalize
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
        folder_out[0],
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
        ax = sns.histplot(
            data_graph['w'],
            bins=max(data_graph['w']),
            kde=True,
            # stat='density',
        )
        plt.ylabel('Frequency')
        plt.xlabel('Original Edge Weight')
        plt.show()

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
        ax = sns.histplot(
            X_scaled,
            kde=True,
            # stat='density',
        )
        plt.ylabel('Frequency')
        plt.xlabel('Scaled Edge Weight')
        plt.show()

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
        graph = nx.read_gpickle(file_in[0])
    else:
        print('File not found')
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
        times = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['times']
        ).iloc[:, 0]
        # Change type (str -> datetime)
        times = times.astype('datetime64[ns]')
    else:
        print('File not found')
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
        nodes = pd.read_csv(
            file_in[0], index_col=False, header=None, names=['nodes']
        ).iloc[:, 0]
    else:
        print('File not found')
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
    output_network=True,
    save_network_csv=True,
    save_network_file=True,
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
    output_network=True,
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
        'bt_ton_network.gpickle',
        'bt_ton_edgelist.csv',
        'bt_ton_delta.csv',
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
    output_delta=True,
    output_network=True,
    save_delta=True,
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
    label_graph='',
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
    graph_name = 'Temporal Network'
    if len(label_graph) > 0: graph_name = graph_name + ' ' + label_graph
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
    # N = file_line_count(path_edit(folder_in,[file_in[1]],label_folder_in,label_file_in,)[0])
    N = len(nodes)

    # Timestamp
    times = temporal_bt_times_read(
        folder_in,
        [file_in[2]],
        label_folder_in,
        label_file_in,
    )
    # T = file_line_count(path_edit(folder_in,[file_in[2]],label_folder_in,label_file_in,)[0])
    T = len(times)

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
        'bt_temporal_times.csv',
        'bt_ton_network.gpickle',
    ],
    label_folder_in='',
    label_file_in='',
    output=False,
    plot=True,
    pw=True,
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

    # Sum of out degrees over nodes
    out_degrees_sum = time_out_degrees.sum(1)

    if output:
        print('Sum of outdegress over nodes')
        print(out_degrees_sum)

    if plot:
        plt.figure()
        ax = sns.histplot(
            out_degrees_sum,
            # bins=max(out_degrees_sum),
            binwidth=100,
            kde=True,
            # stat='density',
        )
        plt.ylabel('Frequency')
        plt.xlabel('Sum of nodes outdegree')
        plt.show()

    # Sum of out degrees over times
    out_degrees_sum = time_out_degrees.sum(0)

    if output:
        print('Distribution of sum of outdegress over timestamp')
        print(out_degrees_sum.value_counts(normalize=True))

    if plot:
        plt.figure()
        ax = sns.histplot(
            out_degrees_sum,
            # bins=max(out_degrees_sum),
            # binwidth=10,
            kde=True,
            # stat='density',
        )
        plt.ylabel('Frequency')
        plt.xlabel('Sum of timestamps outdegree')
        plt.show()

    # Powerlaw correlation of sum of outdegree of nodes over time
    pl = powerlaw.Fit(out_degrees_sum)
    R, p = pl.distribution_compare('power_law', 'lognormal')

    if pw:
        print('PowerLaw Alpha:', pl.power_law.alpha)
        print('PowerLaw Xmin:', pl.power_law.xmin)
        print(R, p)


# ------------
# Edge Weights
# ------------


def edge_weight(
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
    save_weights=True,
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
    N = len(nodes)

    # Timestamp
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
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
            # bins=max(ls2),
            binwidth=0.05,
            binrange=(0, 1),
            # kde=True,
            # stat='density',
        )
        plt.ylabel('Frequency')
        plt.xlabel('Horizontal Edge Weight')
        plt.show()

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
    N = len(nodes)

    # Timestamp
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
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
    # Start with edge weights dictionary
    # Then multiply crossed weight epsilon to weight
    # Scale down weight of horizontal to range [0-0.5], if larger than crossed
    # Scaling formula
    # X_scaled = (b - a) * ((X_scaled - min(X_scaled)) / (max(X_scaled) - min(X_scaled))) + a
    # Which in for range [0-0.5] is as follows
    # X_scaled = (0.5 * np.array(X_scaled)) + 0.5

    for u, v, data in graph.edges(data=True):
        parent_u = u % N
        parent_v = v % N
        time_delta = abs(v - u) // N
        if parent_u != parent_v:
            # Crossed
            # By default epsilon = 1
            ew[(u, v)] = epsilon * data['w']
        # Leave the horizontals as they are ...

    for n in graph:
        parent_n = n % N
        degree_n = graph.out_degree(n)
        w_c = {}  # Weights of crossed edges from n
        w_h = {}  # Weights of horizontal edges from n
        w_all = {}
        for s in sorted(graph.successors(n)):  # N -> S
            parent_s = s % N
            if parent_n != parent_s:  # Crossed
                # We read the weight of edge coming from aggregated network
                w_c[(n, s)] = (ew[(n, s)])
            else:  # Horizontal
                w_h[(n, s)] = (ew[(n, s)])
        # If more than just one horizontal or crossed
        # Max scale the weights
        if len(w_c) + len(w_h) > 1:
            w_h_m = 0
            if len(w_h) > 0:
                w_h_m = max(w_h.values())
            w_c_m = 0
            if len(w_c) > 0:
                w_c_m = max(w_c.values())
            else:
                w_c_m = w_h_m
            # Adjust large weights of horizontal
            # We want horizontal be as important as crossed (no more)
            if w_h_m > w_c_m:
                for key, value in w_h.items():
                    if value > w_c_m:
                        # (1)
                        w_h[key] = w_c_m
                        # (2)
                        # Scale down to a random value [0.5,1]
                        w_h[key] = 0.5 * np.random.sample() + 0.5
            # Normalize node's probabilities
            w_all = w_c.copy()
            w_all.update(w_h)
            # (1)
            # factor = max([w_h_m, w_c_m])
            # w_all = {k: v / factor for k, v in w_all.iteritems()}
            # w_all = {k: v / (factor * degree_n) for k, v in w_all.iteritems()}
            # (2)
            factor = 1.0 / sum(w_all.itervalues())
            w_all = {k: v * factor for k, v in w_all.iteritems()}
            # Update probabilities
            ew.update(w_all)

    if save_probs:
        # pd.DataFrame.from_dict(ew, orient='index').to_csv(file_out[1])
        pd.Series(ew).reset_index().to_csv(
            file_out[1],
            header=False,
            index=False,
        )

    if output_probs:
        for e, w in sorted(ew.items(), key=lambda x: x[0]):
            if e[0] % N == e[1] % N:  # H
                if graph.in_degree(e[1]) == nid:
                    print('{}\t->\t{}\t{}'.format(e[0], e[1], w))

    if plot_probs:
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

    return prob


def ew_read(
    folder_in=NETWORK,
    file_in=['bt_ton_weights.csv'],
    label_folder_in='',
    label_file_in='',
):
    """
    Read edge weights of TON graph
    """
    # Edit paths
    file_in = path_edit(
        file_in,
        folder_in,
        label_file_in,
        label_folder_in,
    )
    # Read edge weights file
    ew = pd.read_csv(file_in[0], names=['u', 'v', 'w'])
    # Fix index from tuple of (u,v)
    ew.index = list(zip(ew.u, ew.v))
    # Only keep the weights
    ew = ew['w']
    # Convert to dict
    ew = ew.to_dict()
    return ew


def prob_read(
    folder_in=NETWORK,
    file_in=['bt_ton_probs.csv'],
    label_folder_in='',
    label_file_in='',
):
    """
    Read edge probabilities of TON graph
    """
    # Edit paths
    file_in = path_edit(
        file_in,
        folder_in,
        label_file_in,
        label_folder_in,
    )
    # Read edge weights file
    prob = pd.read_csv(file_in[0], names=['u', 'v', 'w'])
    # Fix index from tuple of (u,v)
    prob.index = list(zip(ew.u, ew.v))
    # Only keep the weights
    prob = prob['w']
    # Convert to dict
    prob = prob.to_dict()
    return prob


# -------------
# Temporal HITS
# -------------


def hits(
    folder_in=NETWORK,
    folder_out=HITS,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
    ],
    file_out=['a.csv', 'h.csv'],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    nodes=None,
    times=None,
    ew=None,
    nstart=None,
    version=0,
    sigma=0.85,
    max_iter=100,
    tol=1.0e-8,
    norm_max=True,
    norm_final_l1=True,
    norm_final_l2=False,
    norm_iter=False,
    norm_degree=False,
    norm_damping=False,
    norm_scale=False,
    netx_hits=True,
    output=False,
    plot=False,
    save=True,
):
    """
    Calculate HITS centrality of time-ordered network (TON)
    
    Parameters
    ----------
    version : int
        (1) NetworkX
            Normalize scores with SUM=1 Range [0,1]
            1/max normalization in each iteration and final normalization at the end
        (2) Book
            Normalize scores with SUM=1 and DEVIATION=1 and range=[0,1]
        (3) Paper (randomization or teleportation)
            No regular normalization (v1 & v2), just in-out-normalization & damping/teleport/randomize
        (4) NetX + teleport or PARTIAL randomization (NO in-out-normalization)
        (5) Book + teleport or PARTIAL randomization (NO in-out-normalization)
        (6) Paper + normalization of scores at each iteration
        (7) Paper + l2 normalization of score at the end
        (0) Default -> no change to parameters (norm_*) -> version 1 (NetX) + other manually set parameters
    
    Returns
    -------
    dict , dict
        authority and hub scores as {node:score}
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Finishing iteration
    finish_iter = max_iter

    h = {}  # Hub scores
    a = {}  # Authority scores

    # Read graph
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
    N = len(nodes)

    # Times
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
    T = len(times)

    # Edge weight
    if ew is None:
        ew = ew_read(folder_in, [file_in[3]], label_folder_in, label_file_in)

    # Damping-coefficient or normalization-parameter (default = 0.85)
    # Which is same as sigma in supra-matrix model and applies to crossed edges only
    # But here applies to all of the edges (crossed and horizontals ...)
    damping = 1 - sigma  # Default = 0.15

    # NetX
    if version == 1:
        norm_max = True  #
        norm_final_l1 = True  #
        norm_final_l2 = False
        norm_iter = False
        norm_degree = False
        norm_damping = False

    # Book
    if version == 2:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = False
        norm_iter = True  #
        norm_degree = False
        norm_damping = False

    # Paper
    # Kendall shows 3 is most similar to 8 (0.99)
    # Which 8 is the L2 normalized version of 3
    if version == 3:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = False
        norm_iter = False
        norm_degree = True  #
        norm_damping = True  #

    # NetX + teleport
    # Scores are in an exteremly small range and close to 0
    # Kendall shows 4 is most similar to 5 (tau = 1)
    # Alos similar to version 9 (= scaled version of 5) with  tau 1
    if version == 4:
        norm_max = True  #
        norm_final_l1 = True  #
        norm_final_l2 = False
        norm_iter = False
        norm_degree = False
        norm_damping = True  #

    # Book + teleport
    # Score distrobution is more unifor than all (including 4)
    # Since, similar to 4, maybe better to use 5 ?
    if version == 5:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = False
        norm_iter = True  #
        norm_degree = False
        norm_damping = True  #

    # Paper (- norm_max) + NetX = Paper + L1
    # Both norm_max and norm_degree has the same effect
    # So we can (optionally) disable norm_max and kept L1 from NetX
    # Similar to 3 (in terms of distrobutions) However ...
    # Kendall shows 6 is most similar to 7 (0.98) after that 8 (0.96)
    if version == 6:
        norm_max = True  #
        norm_final_l1 = True  #
        norm_final_l2 = False
        norm_iter = False
        norm_degree = True  #
        norm_damping = True  #

    # Paper + Book
    # Converge fast
    # Distribution wise unifor with saturation near end
    # Kendalls show 7 is most similar to 6, eventhough dist are different
    # After that 3 and 8 comes next both with tau 0.97
    if version == 7:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = False
        norm_iter = True  #
        norm_degree = True  #
        norm_damping = True  #

    # Paper + L2
    # Distributiuon is similar to 3 & 6 but in different range
    # Kendalls show 8 is similar to 7 and 6
    if version == 8:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = True  #
        norm_iter = False
        norm_degree = True  #
        norm_damping = True  #

    # Book + teleport + finall scaling
    # Special form of 5 -> 5 + scale [0,1]
    # More comparable with 3
    if version == 9:
        norm_max = False
        norm_final_l1 = False
        norm_final_l2 = False
        norm_iter = True  #
        norm_degree = False
        norm_damping = True  #
        norm_scale = True  #

    # Overally interesting ones are: 3,5,7,9
    # Both 1 & 2 are very similar to 4,5 & 9 -> maybe avoid them ?
    # 4,5,9 are significatly similar (almost same)
    # So 9 makes more sense to use because of it similar range [0,1] just like 3
    # Finally, 6/7 and 3/8 are similar if we want to picl 3 we can do
    # 3 and 9 (maybe 6 as well ?)
    # If narrow to 2 option -> (3) damp and degree_norm (9) damp and general (L2) norm

    # Initialize scores
    if nstart is None:
        h = dict.fromkeys(graph, 1.0 / graph.number_of_nodes())
    else:
        h = nstart
        # Normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s

    # Set start time
    start_time = timeit.default_timer()

    # Power iteration
    if output: print(f'Calculating HITS (Version {version}) ...')
    for i in range(max_iter):
        # Print elapsed time
        if i != 0 and i % 10 == 0:
            elapsed = timeit.default_timer() - start_time
            if output: print(f'Iteration {i} completed {elapsed:.0f} seconds')
            # Reset start time
            start_time = timeit.default_timer()

        # Save the last calculated hub score
        hlast = h

        # Initialize scores for new iteration
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)

        # Authority
        # ---------

        # Left multiply a^T=hlast^T*G
        # Authority of neighbors get value from hub of current node, influenced by weight of edge

        # (1) default HITS -> no degree normalization
        if not norm_degree:
            for n in h:
                for nbr in graph[n]:
                    a[nbr] += hlast[n] * ew.get((n, nbr), 1)
        # (2) degree normalization
        else:
            for n in h:
                for nbr in graph[n]:
                    # a[nbr] += hlast[n] / graph.in_degree(nbr) * ew.get((n, nbr), 1)
                    a[nbr] += hlast[n] * ew.get((n, nbr),
                                                1) / graph.in_degree(nbr)

        # Paper
        # Apply damping factor OR random-walk OR teleport
        if norm_damping:
            for n in a:
                a[n] *= sigma
                a[n] += damping

        # Book
        # Normalized authority scores over root of sum of squares after each calculation
        # In this case we dont need final normalization
        if norm_iter:
            s = 1.0 / math.sqrt(sum([x**2 for x in a.values()]))
            for n in a:
                a[n] *= s

        # Hub
        # ---

        # Multiply h=Ga
        # Hub of current node get value from authority values of neighbor nodes, influenced by weight of edge

        # (1) default HITS -> no degree normalization
        if not norm_degree:
            for n in h:
                for nbr in graph[n]:
                    h[n] += a[nbr] * ew.get((n, nbr), 1)
        # (2) degree normalization
        else:
            for n in h:
                for nbr in graph[n]:
                    # h[n] += a[nbr] / graph.out_degree(n) * ew.get((n, nbr), 1)
                    h[n] += a[nbr] * ew.get((n, nbr), 1) / graph.out_degree(n)

        # Paper
        # Apply damping factor OR randomize
        if norm_damping:
            for n in h:
                h[n] *= sigma
                h[n] += damping

        # Book
        # Normalized hub scores over root of sum of squares after each calculation
        # In this case we dont need final normalization
        if norm_iter:
            s = 1.0 / math.sqrt(sum([x**2 for x in h.values()]))
            for n in h:
                h[n] *= s

        # END of one iteration

        # NetX
        # Normalize scores over maximum
        # Stopping score from getting really large
        if norm_max:
            a_max = 1.0 / max(a.values())
            h_max = 1.0 / max(h.values())
            for n in a:  # OR for n in h => both are same
                a[n] *= a_max
                h[n] *= h_max

        # Check convergence
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            finish_iter = i
            if output: print(f'Successful after {finish_iter} iteration')
            break

    # Program did not meet treshhold for convergence
    if finish_iter == max_iter:
        if output: print(f'Not converged after {finish_iter} iterations')

    # Scale values, MAX = 1
    if norm_scale:
        a_max = 1.0 / max(a.values())
        h_max = 1.0 / max(h.values())
        for n in a:  # OR for n in h => both are same
            a[n] *= a_max
            h[n] *= h_max

    # NetX
    # Last normalization (L1) using sum
    # Output in range (0-1) with sum-all = 1
    if norm_final_l1:
        a_sum = 1.0 / sum(list(a.values()))
        h_sum = 1.0 / sum(list(h.values()))
        for n in a:  # OR for n in h => both are same
            a[n] *= a_sum
            h[n] *= h_sum

    # Last normalization (L2) using sum squred
    if norm_final_l2 and not norm_final_l1:
        a_sum = 1.0 / math.sqrt(sum([x**2 for x in a.values()]))
        h_sum = 1.0 / math.sqrt(sum([x**2 for x in h.values()]))
        for n in a:  # OR for n in h => both are same
            a[n] *= a_sum
            h[n] *= h_sum

    if version not in np.arange(1, 10, 1) and netx_hits:
        # Different options for networkX original HITS calculating
        # (1)
        # h, a = nx.hits(graph)
        # (2)
        # h, a = nx.hits_numpy(graph)
        # (3)
        # h, a = nx.hits_scipy(graph)
        # (4)
        # Adjacency matrix
        A = nx.to_numpy_matrix(graph, dtype=int, weight=None)
        # Eigenvalues and eigenvectors
        evals, evecs = linalg.eig(A.T)
        # Largest eigenvector or PageRank
        # (1)
        # left_vec = evecs[:, 0].T
        # (2)
        left_vec = evecs[:, evals.argmax()].T
        # Normalizing scores
        # left_vec /= left_vec.sum()

    # Save
    if save:
        pd.DataFrame.from_dict(a, orient='index'
                               ).to_csv(file_out[0], header=False)
        pd.DataFrame.from_dict(h, orient='index'
                               ).to_csv(file_out[1], header=False)

    # Plot distribution of scores
    if plot:
        # In most cases, A & H has same value so ploting one of them is enough
        ls_a = sorted(a.values())  # ls_h = sorted(h.values())
        # ax = sns.displot(ls_a)
        # ax = sns.displot(ls_a, height=4, aspect=5)
        ax = sns.displot(ls_a, kde=True, rug=True, height=4, aspect=2.5)
        ax.set_xlabels('HITS Score')

    return a, h


def hits_read(
    folder_in=HITS,
    file_in=['a.csv', 'h.csv'],
    label_folder_in='',
    label_file_in='',
    output=False,
    plot=False
):
    """
    Read the result of HITS centrality scores from file
    """
    # Edit paths
    file_in = path_edit(
        file_in,
        folder_in,
        label_file_in,
        label_folder_in,
    )
    # Read scores from file
    if output: print('Reading HITS score ...')
    a = pd.read_csv(file_in[0], names=['a'], index_col=0)
    h = pd.read_csv(file_in[1], names=['h'], index_col=0)
    # Take column a / h and convert to dict
    a = a.to_dict()['a']
    h = h.to_dict()['h']
    # Plot
    if plot:
        # TODO: needs to be fixed
        # ax = sns.displot(a.values())
        # ax = sns.displot(a.values(), kde=True)
        ax = sns.displot(a.values(), kde=True, rug=True)
    return a, h


def hits_conditional(
    folder_in=[NETWORK, HITS],
    folder_out=HITS,
    file_in=[
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'a.csv',
        'h.csv',
    ],
    file_out=[
        'a_array.csv',
        'h_array.csv',
        'a_avg_node.csv',
        'a_avg_time.csv',
        'h_avg_node.csv',
        'h_avg_time.csv',
        'a_norm_node.csv',
        'a_norm_time.csv',
        'h_norm_node.csv',
        'h_norm_time.csv',
    ],
    label_folder_in=['', ''],
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    a=None,
    h=None,
    N=None,
    T=None,
    removed=False,
    save=True
):
    """
    Create conditional centralities by normalizing
    or averaging HITS scores over time (column) or node (row)
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Nodes
    if N is None:
        N = file_line_count(
            path_edit(
                [file_in[0]],
                folder_in[0],
                label_file_in,
                label_folder_in[0],
            )[0]
        )

    # Times
    if T is None:
        T = file_line_count(
            path_edit(
                [file_in[1]],
                folder_in[0],
                label_file_in,
                label_folder_in[0],
            )[0]
        )

    # Read HITS scores
    if a is None or h is None:
        a, h = hits_read(
            folder_in[1],
            file_in[2:4],
            label_folder_in[1],
            label_file_in,
        )

    # Convert score dict to matrix (N x T)
    A = np.zeros((N, T + 1))
    H = np.zeros((N, T + 1))
    # If graph is complete (no node has been removed)
    if not removed:
        a_sorted = [a[k] for k in sorted(a.keys())]
        h_sorted = [h[k] for k in sorted(h.keys())]
        A = np.reshape(a_sorted, (T + 1, N)).T
        H = np.reshape(h_sorted, (T + 1, N)).T
    else:  # Some of the nodes were removed
        for node in a.keys():  # h.keys()
            row = node % N  # node
            col = node // N  # time
            A[row, col] = a[node]
            H[row, col] = h[node]

    # Save A & H matrices
    if save:
        pd.DataFrame(A).to_csv(file_out[0], header=None, index=None)
        pd.DataFrame(H).to_csv(file_out[1], header=None, index=None)

    # Read A & H matrices
    # A = pd.read_csv(file_input[0]).values
    # H = pd.read_csv(file_input[1]).values

    # Conditional Centralities
    # ------------------------

    # Average of score over time (column) or node (row)
    A_avg_node = np.mean(A, axis=1)
    A_avg_time = np.mean(A, axis=0)
    H_avg_node = np.mean(H, axis=1)
    H_avg_time = np.mean(H, axis=0)

    # Normalized scores over time (column) or node (row)
    # L1 is over sum(abs(x)) but scince all values are + L1 is over sum(x)
    A_norm_node = normalize(A, axis=1, norm='l1')
    A_norm_time = normalize(A, axis=0, norm='l1')
    H_norm_node = normalize(H, axis=1, norm='l1')
    H_norm_time = normalize(H, axis=0, norm='l1')
    # OR
    # A_norm_node = A / A.sum(axis=1, keepdims=True)
    # A_norm_time = A / A.sum(axis=0, keepdims=True)
    # H_norm_node = H / H.sum(axis=1, keepdims=True)
    # H_norm_time = H / H.sum(axis=0, keepdims=True)

    if save:
        pd.DataFrame(A_avg_node).to_csv(file_out[2], header=None, index=None)
        pd.DataFrame(A_avg_time).to_csv(file_out[3], header=None, index=None)
        pd.DataFrame(H_avg_node).to_csv(file_out[4], header=None, index=None)
        pd.DataFrame(H_avg_time).to_csv(file_out[5], header=None, index=None)
        pd.DataFrame(A_norm_node).to_csv(file_out[6], header=None, index=None)
        pd.DataFrame(A_norm_time).to_csv(file_out[7], header=None, index=None)
        pd.DataFrame(H_norm_node).to_csv(file_out[8], header=None, index=None)
        pd.DataFrame(H_norm_time).to_csv(file_out[9], header=None, index=None)


def hits_conditional_read(
    folder_in=HITS,
    file_in=[
        'a_array.csv',
        'h_array.csv',
        'a_avg_node.csv',
        'a_avg_time.csv',
        'h_avg_node.csv',
        'h_avg_time.csv',
        'a_norm_node.csv',
        'a_norm_time.csv',
        'h_norm_node.csv',
        'h_norm_time.csv',
    ],
    label_folder_in='',
    label_file_in='',
    return_matrices=[0, 1],
    return_all=False,
    output=False
):
    """
    Read conitional HITS centrality scores
    
    Parameters
    ----------
    return_matrices : list
        [0,1] -> return matrices A & H only
        range(0,10) -> return all conditional matrices
    """
    # Edit paths
    file_in = path_edit(
        file_in,
        folder_in,
        label_file_in,
        label_folder_in,
    )

    # Return object names
    rtn = {}
    rtn_names = [
        'A', 'H', 'a_avg_node', 'a_avg_time', 'h_avg_node', 'h_avg_time',
        'a_norm_node', 'a_norm_time', 'h_norm_node', 'h_norm_time'
    ]

    # Default only return (0) A and (1) H matrices
    if return_all:
        return_matrices = list(range(0, 10))

    # Read the files
    if output: print('Readinging HITS conditional ...')
    for i in return_matrices:
        if output: print(f'{rtn_names[i]}')
        temp = pd.read_csv(file_in[i], header=None, index_col=False).values
        rtn[rtn_names[i]] = temp

    return rtn


def hits_analyze(
    folder_in=[NETWORK, HITS],
    folder_out=HITS,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
        'a.csv',
        'h.csv',
        'a_array.csv',
        'h_array.csv',
        'a_avg_node.csv',
        'a_avg_time.csv',
        'h_avg_node.csv',
        'h_avg_time.csv',
        'a_norm_node.csv',
        'a_norm_time.csv',
        'h_norm_node.csv',
        'h_norm_time.csv',
    ],
    file_out=[
        'report.csv',
        'top.csv',
        'fig_a.pdf',
        'fig_h.pdf',
        'fig_a_report.pdf',
        'fig_h_report.pdf',
        'fig_a_corr.pdf',
        'fig_h_corr.pdf',
        'fig_mat.pdf',
    ],
    label_folder_in=['', ''],
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    nodes=None,
    times=None,
    ew=None,
    a=None,
    h=None,
    top=2,
    section=4,
    report_num=-1
):
    """
    Analyze HITS scores
        Find top rank nodes using averaged-score over time
        Highlight top nodes of TON model and other info such as parent, time, in-out-degree, in-out-weight ...
    
    Parameters
    ----------
        a : dict
            authority score {node:score}
        h : dict
            hub score {node:score}
        graph: NetX
            time-ordered network (TON) model
        times : list
        nodes : list
        ew : dict
            edge weights {(u,v):w}
    """
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Graph
    if graph is None:
        graph = ton_bt_read(
            folder_in[0],
            [file_in[0]],
            label_folder_in[0],
            label_file_in,
        )

    # Nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in[0],
            [file_in[1]],
            label_folder_in[0],
            label_file_in,
        )
    N = len(nodes)

    # Times
    if times is None:
        times = temporal_bt_times_read(
            folder_in[0],
            [file_in[2]],
            label_folder_in[0],
            label_file_in,
        )
    times = list(times)
    T = len(times)

    # Edge weight
    if ew is None:
        ew = ew_read(
            folder_in[0],
            [file_in[3]],
            label_folder_in[0],
            label_file_in,
        )

    # # Read HITS
    if a is None or h is None:
        a, h = hits_read(
            folder_in[1],
            file_in[4:6],
            label_folder_in[1],
            label_file_in,
        )

    times.insert(0, times[0])
    t_first = times[0].strftime('%d %B\n%I %p')
    t_last = times[-1].strftime('%d %B\n%I %p')
    t_0 = times[0] - pd.Timedelta(1, unit='h')
    t_day_idx = []
    t_day_week = []
    t_day_date = []
    t_hour_12 = []
    t_hour_17 = []
    for i, time in enumerate(times):
        if t_0.weekday() != time.weekday():
            t_day_idx.append(i)
            t_day_week.append(time.strftime('%a'))
            t_day_date.append(time.strftime('%d'))
            t_0 = time
        if time.strftime('%H') == '12':
            t_hour_12.append(i)
        if time.strftime('%H') == '17':
            t_hour_17.append(i)

    # Conditional HTIS scores
    cs = hits_conditional_read(
        folder_in[1],
        file_in[6:16],
        label_folder_in[1],
        label_file_in,
        return_all=True,
    )

    # Authority Plot
    # --------------

    fig, ax = plt.subplots(figsize=(32, 20))
    # (1)
    # norm = mpl.colors.Normalize(vmin=0, vmax=1.0, clip=True)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlGnBu)
    # (2)
    # mapper = cm.get_cmap('YlGnBu', 10)  # 10 discrete colors
    # force the first color entry to be grey
    # (3)
    cmap = plt.cm.tab10_r  # Define the colormap (tab10_r, YlGnBu, Wistia, ocean, cool)
    # Extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # If tab10
    cmaplist[0], cmaplist[2] = cmaplist[2], cmaplist[0]
    # Force the first color entry to be grey
    cmaplist[0] = (.8, .8, .8, 1.0)
    # Create the new custom map
    mapper = mpl.colors.LinearSegmentedColormap.from_list(
        'my cmap', cmaplist, cmap.N
    )
    # define the bins and normalize
    bounds = np.linspace(0, 1, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # We can modify and add removed nodes or score == 0
    # As red dots or something like that in the plot, but we need to
    # Know if score == 0 is considered as removed or we get a removed_list
    # Of nodes as an input of the hits_analyze method

    A = cs['A']
    X = list(range(1, T + 2))
    for row in range(len(A)):
        sc = ax.scatter(
            X,
            [row + 1] * (T + 1),
            s=A[row] * 1000,
            # s=[4*x**2 for x in A[row] * 10],
            c=A[row],
            # c=[mapper.to_rgba(x) for x in A[row]],
            marker='|',
            alpha=0.8,
            # vmin=0,
            # vmax=1.0,
            norm=norm,  # or use vmin/vmax
            # cmap=cm.YlGnBu,
            cmap=mapper,
        )

    A_avg_node = cs['a_avg_node']
    ax.scatter(
        [T + 3] * N,
        list(range(1, N + 1)),
        s=A_avg_node * 100,
        c=A_avg_node,
        marker='s',
        alpha=0.5,
        cmap=cm.Wistia,
    )

    A_avg_node_sort = sorted(A_avg_node, reverse=True)
    A_avg_node_rank = [A_avg_node_sort.index(e) for e in A_avg_node]
    for x, y in zip([T + 5] * N, list(range(1, N + 1))):
        ax.text(
            x,
            y,
            str(A_avg_node_rank[y - 1] + 1),
            color='black',
            fontsize=10,
        )

    A_avg_time = cs['a_avg_time']
    ax.scatter(
        X,
        [N + 1] * (T + 1),
        s=A_avg_time * 100,
        c=A_avg_time,
        marker='s',
        alpha=0.5,
        cmap=cm.Wistia,
    )

    # Vertical date lines
    ax.text(0, -0.2, 'Mon', rotation=90)
    for i, point in enumerate(t_day_idx):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='green',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, t_day_week[i], rotation=90)
    for i, point in enumerate(t_hour_12):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='red',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, 'Noon', rotation=90, fontsize=6)
    for i, point in enumerate(t_hour_17):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='blue',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, '5 PM', rotation=90, fontsize=6)

    # X axes
    ax.set_xlabel('Time')
    # ax.set_xticks(range(0, T + 1, 10))
    # OR
    ax.set_xticks([1] + t_day_idx + [T + 1])
    ax.set_xticklabels([t_first] + t_day_date + [t_last])
    # ax_label_first = ax.get_xticklabels()[0]
    # ax_label_first.set_rotation(45)
    # ax_label_first.set_ha('left')
    # ax_label_last = ax.get_xticklabels()[-1]
    # ax_label_last.set_rotation(45)
    # ax_label_last.set_ha('left')

    # Y axes
    ax.set_ylabel('Node Index')
    ax.set_yticks(range(1, N + 1))

    # Figure labels
    fig_title = 'Authority Score Over Time'
    if len(label_file_out) > 0:
        fig_title = fig_title + ' (' + label_file_out + ')'
    ax.set_title(fig_title)

    # Color bar
    # fig.colorbar(mapper, ax=ax, shrink=0.5, pad=0.01)
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)

    # Save figure
    fig.savefig(file_out[2], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)  # Close the figure window

    # HUB
    # ---

    H = cs['H']
    fig, ax = plt.subplots(figsize=(32, 20))
    for row in range(len(H)):
        ax.scatter(
            X,
            [row + 1] * (T + 1),
            s=H[row] * 1000,
            c=H[row],
            marker='|',
            alpha=0.8,
            norm=norm,
            cmap=mapper,
        )

    H_avg_node = cs['h_avg_node']
    ax.scatter(
        [T + 3] * N,
        list(range(1, N + 1)),
        s=H_avg_node * 100,
        c=H_avg_node,
        marker='s',
        alpha=0.5,
        cmap=cm.Wistia
    )

    H_avg_node_sort = sorted(H_avg_node, reverse=True)
    H_avg_node_rank = [H_avg_node_sort.index(e) for e in H_avg_node]
    for x, y in zip([T + 5] * N, list(range(1, N + 1))):
        ax.text(
            x,
            y,
            str(H_avg_node_rank[y - 1] + 1),
            color='black',
            fontsize=10,
        )

    H_avg_time = cs['h_avg_time']
    ax.scatter(
        X,
        [N + 1] * (T + 1),
        s=H_avg_time * 100,
        c=H_avg_time,
        marker='s',
        alpha=0.5,
        cmap=cm.Wistia,
    )

    ax.text(0, -0.2, 'Mon', rotation=90)
    for i, point in enumerate(t_day_idx):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='green',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, t_day_week[i], rotation=90)
    for i, point in enumerate(t_hour_12):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='red',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, 'Noon', rotation=90, fontsize=6)
    for i, point in enumerate(t_hour_17):
        ax.axvline(
            x=point,
            ymin=0.03,
            ymax=0.95,
            linewidth=0.5,
            color='blue',
            alpha=0.25
        )
        ax.text(point - 1, -0.2, '5 PM', rotation=90, fontsize=6)

    ax.set_xlabel('Time')
    ax.set_xticks([1] + t_day_idx + [T + 1])
    ax.set_xticklabels([t_first] + t_day_date + [t_last])
    ax.set_ylabel('Node Index')
    ax.set_yticks(range(1, N + 1))
    fig_title = 'Hub Score Over Time'
    if len(label_file_out) > 0:
        fig_title = fig_title + ' (' + label_file_out + ')'
    ax.set_title(fig_title)
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
    fig.savefig(file_out[3], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # REPORT
    # -------
    # Top node of temporal graph

    # Check the correctness value of section and top
    if top * section > N or top == -1:
        # Analyze all nodes
        top = N
        section = 1

    # Selected top nodes
    selected_a = []
    selected_h = []
    # Rank nodes based on average score
    idx_a, _ = list(zip(*rank(A_avg_node)))
    idx_h, _ = list(zip(*rank(H_avg_node)))

    # Select top 'n' nodes from each split
    for sp in np.array_split(idx_a, section):
        selected_a.extend(sp[:top])
    for sp in np.array_split(idx_h, section):
        selected_h.extend(sp[:top])

    df_columns = []
    df_labels = ['_a', '_h', '_id', '_od', '_iw', '_ow']
    df_index = []

    for node in selected_a:
        df_index.extend([str(node) + e for e in df_labels])
        col_in_d = []
        col_out_d = []
        col_in_w = []
        col_out_w = []
        for i in range(0, N * (T + 1), N):  # T + 1 iteration
            n = node + i  # Give node index over different times
            if graph.has_node(n):
                col_in_d.append(graph.in_degree(n))
                col_out_d.append(graph.out_degree(n))
                sum_w = 0
                for nbr in graph.predecessors(n):
                    sum_w += ew.get((nbr, n), 1)
                col_in_w.append(sum_w)
                sum_w = 0
                for nbr in graph.successors(n):
                    sum_w += ew.get((n, nbr), 1)
                col_out_w.append(sum_w)
            else:  # The node has been removed from graph
                col_in_d.append(0)
                col_out_d.append(0)
                col_in_w.append(0)
                col_out_w.append(0)
        df_columns.append(A[node])
        df_columns.append(H[node])
        df_columns.append(col_in_d)
        df_columns.append(col_out_d)
        df_columns.append(col_in_w)
        df_columns.append(col_out_w)

    for node in selected_h:
        df_index.extend([str(node) + e for e in df_labels])
        col_in_d = []
        col_out_d = []
        col_in_w = []
        col_out_w = []
        for i in range(0, N * (T + 1), N):
            n = node + i
            if graph.has_node(n):
                col_in_d.append(graph.in_degree(n))
                col_out_d.append(graph.out_degree(n))
                sum_w = 0
                for nbr in graph.predecessors(n):
                    sum_w += ew.get((nbr, n), 1)
                col_in_w.append(sum_w)
                sum_w = 0
                for nbr in graph.successors(n):
                    sum_w += ew.get((n, nbr), 1)
                col_out_w.append(sum_w)
            else:
                col_in_d.append(0)
                col_out_d.append(0)
                col_in_w.append(0)
                col_out_w.append(0)
        df_columns.append(A[node])
        df_columns.append(H[node])
        df_columns.append(col_in_d)
        df_columns.append(col_out_d)
        df_columns.append(col_in_w)
        df_columns.append(col_out_w)

    # Create and save dataframe of top nodes from each percentile (or split)
    df = pd.DataFrame(df_columns, index=df_index, columns=list(range(T + 1))).T
    df.to_csv(file_out[0], index=False)

    # Plot the report
    # df = df.t
    df = pd.read_csv(file_out[0]).T
    times = list(df.columns)
    idx = df.index.values.tolist()
    a_nodes, h_nodes = np.array_split(
        [int(idx[i].split('_')[0]) for i in range(0, len(idx), 6)], 2
    )

    # Authority scores
    fig, axs = plt.subplots(
        len(a_nodes),
        1,
        figsize=(16, 12),
        constrained_layout=True,
    )
    # cmap = plt.get_cmap('tab10')
    for i in range(len(a_nodes)):
        row_label = str(a_nodes[i]) + '_a'
        row_label2 = str(a_nodes[i]) + '_id'  # id,od,iw,ow
        legend_label = str(a_nodes[i])
        # axs[i].scatter(
        axs[i].plot(
            times,
            df.loc[row_label],
            # (0)
            c='black',
            alpha=0.5,
            # (1)
            # c=[np.random.rand(3, )],
            # (2)
            # c=[cmap(i)],
            # (1)
            # linewidth=1,
            # marker='o',
            # markersize=6,
            # (2)
            # marker='.',
            # label=legend_label
        )
        sc = axs[i].scatter(
            times,
            [0] * len(times),
            # df.loc[row_label2],
            s=df.loc[row_label2] * 100,
            c=df.loc[row_label2],
            marker='|',
            alpha=1.0,
            vmin=0,
            # vmin=min(df.loc[row_label2]),
            vmax=28,
            # vmax=max(df.loc[row_label2]),
            cmap=cm.Wistia,
        )
        axs[i].set_ylabel('Node ' + str(a_nodes[i]))
    # fig.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))
    fig.text(0.5, 1.02, 'Temporal HITS', ha='center')
    fig.text(0.5, -0.02, 'Time', ha='center')
    fig.text(-0.02, 0.5, 'Authority Score', va='center', rotation='vertical')
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
    fig.savefig(file_out[4], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Hub scores
    fig, axs = plt.subplots(
        len(h_nodes),
        1,
        figsize=(16, 12),
        constrained_layout=True,
    )
    for i in range(len(h_nodes)):
        row_label = str(h_nodes[i]) + '_h'
        row_label2 = str(h_nodes[i]) + '_od'  # id,od,iw,ow
        axs[i].plot(
            times,
            df.loc[row_label],
            c='black',
            alpha=0.5,
        )
        sc = axs[i].scatter(
            times,
            [0] * len(times),
            s=df.loc[row_label2] * 100,
            c=df.loc[row_label2],
            marker='|',
            alpha=1.0,
            vmin=0,
            vmax=28,
            cmap=cm.Wistia,
        )
        axs[i].set_ylabel('Node ' + str(a_nodes[i]))
    fig.text(0.5, 1.02, 'Temporal HITS', ha='center')
    fig.text(0.5, -0.02, 'Time', ha='center')
    fig.text(-0.02, 0.5, 'Hub Score', va='center', rotation='vertical')
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
    fig.savefig(file_out[5], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # TOP
    # ---
    # Top nodes of TON graph

    if report_num > N * (T + 1) or report_num == -1:
        # Analyze all nodes
        report_num = N * (T + 1)

    # Top rank TON graph nodes based on HITS score
    A_top = rank(a, return_rank=True)[:report_num]
    H_top = rank(h, return_rank=True)[:report_num]

    df_index = []
    col_r = []  # Rank
    col_c = []  # Category i.e. A or H
    col_n = []  # Node id
    col_a = []
    col_h = []
    col_p = []  # Parent
    col_t = []  # Time
    col_in_d = []
    col_out_d = []
    col_in_w = []
    col_out_w = []

    for n, r in A_top.items():
        df_index.append('a_' + str(n))
        col_n.append(n)
        col_r.append(r)
        col_c.append('a')
        col_a.append(a[n])
        col_h.append(h[n])
        col_p.append(n % N)
        col_t.append(n // N)
        col_in_d.append(graph.in_degree(n))
        col_out_d.append(graph.out_degree(n))
        sum_w = 0
        for nbr in graph.predecessors(n):
            sum_w += ew.get((nbr, n), 1)
        col_in_w.append(sum_w)
        sum_w = 0
        for nbr in graph.successors(n):
            sum_w += ew.get((n, nbr), 1)
        col_out_w.append(sum_w)

    for n, r in H_top.items():
        df_index.append('h_' + str(n))
        col_n.append(n)
        col_r.append(r)
        col_c.append('h')
        col_a.append(a[n])
        col_h.append(h[n])
        col_p.append(n % N)
        col_t.append(n // N)
        col_in_d.append(graph.in_degree(n))
        col_out_d.append(graph.out_degree(n))
        sum_w = 0
        for nbr in graph.predecessors(n):
            sum_w += ew.get((nbr, n), 1)
        col_in_w.append(sum_w)
        sum_w = 0
        for nbr in graph.successors(n):
            sum_w += ew.get((n, nbr), 1)
        col_out_w.append(sum_w)

    df = pd.DataFrame(
        {
            'r': col_r,
            'n': col_n,
            'a': col_a,
            'h': col_h,
            'p': col_p,
            't': col_t,
            'id': col_in_d,
            'od': col_out_d,
            'iw': col_in_w,
            'ow': col_out_w,
            'c': col_c
        },
        index=df_index
    )

    # Save top node analysis
    df.to_csv(file_out[1], index=False)

    # Correlation between in/out-degree and HITS scores
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
    axs[0, 0].scatter(df[df.c == 'a'].id, df[df.c == 'a'].a)
    axs[0, 0].set_xlabel('In-degree')
    axs[0, 0].set_ylabel('Authority Score')
    axs[0, 1].scatter(df[df.c == 'a'].od, df[df.c == 'a'].a)
    axs[0, 1].set_xlabel('Out-degree')
    axs[0, 1].set_ylabel('Authority Score')
    axs[1, 0].scatter(df[df.c == 'a'].iw, df[df.c == 'a'].a)
    axs[1, 0].set_xlabel('In-weight')
    axs[1, 0].set_ylabel('Authority Score')
    axs[1, 1].scatter(df[df.c == 'a'].ow, df[df.c == 'a'].a)
    axs[1, 1].set_xlabel('Out-weight')
    axs[1, 1].set_ylabel('Authority Score')
    fig.savefig(file_out[6], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
    axs[0, 0].scatter(df[df.c == 'h'].id, df[df.c == 'h'].h)
    axs[0, 0].set_xlabel('In-degree')
    axs[0, 0].set_ylabel('Hub Score')
    axs[0, 1].scatter(df[df.c == 'h'].od, df[df.c == 'h'].h)
    axs[0, 1].set_xlabel('Out-degree')
    axs[0, 1].set_ylabel('Hub Score')
    axs[1, 0].scatter(df[df.c == 'h'].iw, df[df.c == 'h'].h)
    axs[1, 0].set_xlabel('In-weight')
    axs[1, 0].set_ylabel('Hub Score')
    axs[1, 1].scatter(df[df.c == 'h'].ow, df[df.c == 'h'].h)
    axs[1, 1].set_xlabel('Out-weight')
    axs[1, 1].set_ylabel('Hub Score')
    fig.savefig(file_out[7], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Correlation matrix
    corr = df.corr()  # method='pearson',
    # corr = df.corr(method ='kendall')
    # corr = df.corr(method ='spearman')
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, ax=ax, cmap='YlGnBu', linewidths=0.1)
    fig.savefig(file_out[8], dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


def hits_group(
    folder_in=[NETWORK, HITS],
    folder_out=HITS,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
    ],
    file_out=[
        'a.csv',
        'h.csv',
        'a_array.csv',
        'h_array.csv',
        'a_avg_node.csv',
        'a_avg_time.csv',
        'h_avg_node.csv',
        'h_avg_time.csv',
        'a_norm_node.csv',
        'a_norm_time.csv',
        'h_norm_node.csv',
        'h_norm_time.csv',
    ],
    file_out2=[
        'report.csv',
        'top.csv',
        'fig_a.pdf',
        'fig_h.pdf',
        'fig_a_report.pdf',
        'fig_h_report.pdf',
        'fig_a_corr.pdf',
        'fig_h_corr.pdf',
        'fig_mat.pdf',
    ],
    label_folder_in='',
    # label_folder_out='',
    label_file_in='',
    label_file_out='',
    #
    versions=None,
    WR=False,
    RD=True,
    DIST=False,
    TAU=False,
    #
    ISIM=True,
    rep=1,
    k=0,
    #
    graph=None,
    nodes=None,
    times=None,
    ew=None,
    nstart=None,
    sigma=0.85,
    max_iter=100,
    tol=1.0e-8,
    #
    a=None,
    h=None,
    N=None,
    T=None,
    removed=False,
    #
    top=2,
    section=4,
    report_num=-1,
    #
    output=False,
    save=True,
):
    """
    Perfrom a group or series of HITS
    based on type, if VAR or ISIM ...
    """
    if versions is None:
        versions = list(np.arange(1, 10, 1))  # 1 ... 9
    auts = {}
    hubs = {}
    if WR:
        for v in versions:
            label = 'group/' + str(v)
            os.makedirs(os.path.join(folder_out, label), exist_ok=True)
            if sigma < 0.85 or sigma > 0.85:
                label = label + '-' + str(int(sigma * 100))
            # Hits
            a, h = hits(
                folder_in=folder_in[0],
                folder_out=folder_out,
                file_in=file_in[0:4],
                file_out=file_out,
                label_folder_in=label_folder_in,
                label_folder_out=label,
                label_file_in=label_file_in,
                label_file_out=label_file_out,
                graph=graph,
                nodes=nodes,
                times=times,
                ew=ew,
                nstart=nstart,
                version=v,
                sigma=sigma,
                max_iter=max_iter,
                tol=tol,
                output=output,
                save=save,
            )
            # Save results for comparisions
            auts[v] = a
            hubs[v] = h
            # Conditional
            hits_conditional(
                folder_in=folder_in,
                folder_out=folder_out,
                file_in=[file_in[1], file_in[2], file_out[0], file_out[1]],
                file_out=file_out[2:12],
                label_folder_in=[label_folder_in, label],
                label_folder_out=label,
                label_file_in=label_file_in,
                label_file_out=label_file_out,
                a=a,
                h=h,
                N=N,
                T=T,
                removed=removed,
                save=save,
            )
            # Analyze
            hits_analyze(
                folder_in=folder_in,
                folder_out=folder_out,
                file_in=file_in + file_out,
                file_out=file_out2,
                label_folder_in=[label_folder_in, label],
                label_folder_out=label,
                label_file_in=label_file_in,
                label_file_out=label_file_out,
                graph=graph,
                nodes=nodes,
                times=times,
                ew=ew,
                a=a,
                h=h,
                top=top,
                section=section,
                report_num=-report_num,
            )
    if RD:
        for v in versions:
            label = 'group/' + str(v)
            os.makedirs(os.path.join(folder_out, label), exist_ok=True)
            if sigma < 0.85 or sigma > 0.85:
                label = label + '-' + str(int(sigma * 100))
            # Hits
            auts[v], hubs[v] = hits_read(
                folder_in=folder_in[1],
                file_in=file_out[0:2],
                label_folder_in=label,
                label_file_in=label_file_in,
                output=output,
                plot=False,
            )

    # Kendall tau
    taus = {}
    if TAU:
        for i in versions:
            for j in versions:
                if i != j and (i, j) not in taus and (j, i) not in taus:
                    # arank1 = rank(auts[i])
                    # arank2 = rank(auts[j])
                    arank1 = rank(auts[i],
                                  return_rank=True).sort_index().values
                    arank2 = rank(auts[j],
                                  return_rank=True).sort_index().values
                    tau, apvalue = stats.kendalltau(arank1, arank2)
                    taus[(i, j)] = tau
                    taus[(j, i)] = tau
        atau = np.eye(len(versions))
        # htau = np.eye(len(versions))
        for key in taus:
            atau[key[0] - 1, key[1] - 1] = taus[key]
        # Plot the Kendalls tau
        ax = sns.heatmap(
            tau,
            linewidth=0.1,
            annot=True,
            cmap='YlGnBu',
            xticklabels=range(1, 10),
            yticklabels=range(1, 10),
        )
        ax.invert_yaxis()
        plt.title('Kendall Tau of Different HITS Variations')
        # plt.xlabel('HITS Version')
        # plt.ylabel('HITS Version')
        plt.tight_layout()
        # plt.show()
        plt.savefig('kendall_tau.pdf', dpi=400)

    # Intersection similarity
    isims = {}
    if ISIM:
        for i in versions:
            for j in versions:
                if i != j and (i, j) not in isims and (j, i) not in isims:
                    isims[(i, j)] = isim_hits(
                        auts[i],
                        auts[j],
                        str(i),
                        str(j),
                        # rep=rep,
                        # k=k,
                        rep=1000,
                        k=1000,
                    )
                    isims[(j, i)] = 0

    # Return
    return {
        'hits': (auts, hubs),
        'tau': taus,
        'isim': isims,
    }


# -----------------------
# Intersection Similarity
# -----------------------

# def isim_hits(
#     scores1,
#     scores2,
#     name1='',
#     name2='',
#     rep=1,
#     plot=True,
# ):
#     """
#     Receives two dictionary of {node:score}
#     where one can have a subset of keys from another one {keys1} <= {keys2}
#     First add
#     """
#     uniq1 = len(set(scores1.values())) / len(scores1)
#     uniq2 = len(set(scores2.values())) / len(scores2)
#     print(f'|S1| = {len(scores1)} with {uniq1 * 100:.2f} % unique scores')
#     print(f'|S2| = {len(scores2)} with {uniq2 * 100:.2f} % unique scores')
#     uniqmin = min([uniq1, uniq2])
#     lenmax = max([len(scores1), len(scores2)])
#     # Number of experiment repeat
#     if rep < 1:
#         rep = int(lenmax * (1 - (uniqmin)))
#         print(f'repeat for {rep}')
#     if len(scores1) > len(scores2):
#         rem = set(scores1).difference(set(scores2))
#         for key in rem:
#             scores2[key] = 0
#     elif len(scores1) < len(scores2):
#         rem = set(scores2).difference(set(scores1))
#         for key in rem:
#             scores1[key] = 0
#     # List of nodes index sorted based on rank low-to-high or score high-to-low
#     r1 = breakdown(scores1, 0, True)[0]
#     r2 = breakdown(scores2, 0, True)[0]
#     isims = defaultdict(list)
#     for r in range(rep):
#         isim = []
#         for i in range(1, len(r1) + 1):
#             s1 = set(r1[:i])
#             s2 = set(r2[:i])
#             dif = s1 ^ s2
#             isim.append(len(dif) / (2 * i))
#         for i in range(len(isim)):
#             isims[i].append(sum(isim[:i + 1]) / (i + 1))

#     # Covert all result to dataframe for futher visualization
#     df = pd.DataFrame.from_dict(isims)
#     if plot:
#         fig, ax = plt.subplots()
#         df.mean().plot(title='Intersection Similarity', ax=ax)
#         fig_name = 'isim' + '-' + name1 + '-' + name2 + '.pdf'
#         plt.title('Intersection Similarity ' + name1 + '-' + name2)
#         plt.tight_layout()
#         # plt.show()
#         plt.savefig(fig_name, dpi=400)
#     return df


def isim_hits(
    scores1,
    scores2,
    name1='',
    name2='',
    rep=1,
    k=0,
    output=True,
    plot=True,
):
    """
    Second version of intersection similarity for HITS scores from two observations
    the difference is that, we assume second observation comes from graph where some
    of high rank nodes are being removed the second list is smaller than first
    despite version 1, we don't add score 0 for removed nodes, but we only consider
    a smaller version of larger set equalt to smaller set so that both have same size
    and ISIM is only being calculated on common set of nodes, therefore the number of
    K as maximum size of ISIM would be based on intersection set
    """
    uniq1 = len(set(scores1.values())) / len(scores1)
    uniq2 = len(set(scores2.values())) / len(scores2)
    if output:
        print(
            f'|S{name1}| = {len(scores1)} with {uniq1 * 100:.2f} % unique scores'
        )
        print(
            f'|S{name2}| = {len(scores2)} with {uniq2 * 100:.2f} % unique scores'
        )
        print()
    uniqmin = min([uniq1, uniq2])
    lenmax = max([len(scores1), len(scores2)])
    # Number of experiment repeat
    if rep < 1:
        rep = int(lenmax * (1 - (uniqmin)))
        print(f'repeat for {rep}')
    # Intersection of keys of two score dictionary
    com = set(scores1).intersection(set(scores2))
    # Only consider scores of common nodes
    if len(scores1) > len(scores2):
        scores1 = {key: scores1[key] for key in com}
    elif len(scores1) < len(scores2):
        scores2 = {key: scores2[key] for key in com}
    # Only consider top K nodes in list if K is initializem, other wise => size(nodes)
    rtn = False
    if k < 1:
        rtn = True
    else:
        # Check k is not larger than size(nodes)
        if k > len(scores1):
            k = len(scores1)
    # Intersection similarity of two rank set
    isims = defaultdict(list)
    for r in range(rep):
        isim = []
        # List of nodes sorted based on low-to-high rank = high-to-low score
        r1 = breakdown(scores1, k, rtn)[0]
        r2 = breakdown(scores2, k, rtn)[0]
        for i in range(1, len(r1) + 1):
            s1 = set(r1[:i])
            s2 = set(r2[:i])
            dif = s1 ^ s2
            isim.append(len(dif) / (2 * i))
        for i in range(len(isim)):
            isims[i].append(sum(isim[:i + 1]) / (i + 1))
    # Covert all result to dataframe for futher visualization
    df = pd.DataFrame.from_dict(isims)
    if plot:
        fig, ax = plt.subplots()
        # (1)
        # df.mean().plot(ax=ax)
        # (2)
        X = np.arange(1, k + 1)
        Y_min = df.min(axis=0)
        Y_max = df.max(axis=0)
        Y_mean = df.mean(axis=0)
        ax.plot(X, Y_mean)
        ax.fill_between(X, Y_max, Y_min, alpha=0.5)
        if len(name1) > 0 and len(name2) > 0:
            fig_name = 'isim' + '-' + name1 + '-' + name2 + '.pdf'
            plt.title('Intersection Similarity ' + name1 + '-' + name2)
        else:
            fig_name = 'isim.pdf'
            plt.title('Intersection Similarity')
        plt.tight_layout()
        plt.savefig(fig_name, dpi=400)
        # plt.show()
    return df


# ------------
# Node Removal
# ------------


def hits_remove(
    folder_in=[NETWORK, HITS],
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
        'a.csv',
        'h.csv',
    ],
    file_out=[
        'remove.csv',
        'bt_ton_network.gpickle',
        'bt_temporal_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
        'bt_ton_probs.csv',
        'a.csv',
        'h.csv',
        'a_array.csv',
        'h_array.csv',
        'a_avg_node.csv',
        'a_avg_time.csv',
        'h_avg_node.csv',
        'h_avg_time.csv',
        'a_norm_node.csv',
        'a_norm_time.csv',
        'h_norm_node.csv',
        'h_norm_time.csv',
        'report.csv',
        'top.csv',
        'fig_a.pdf',
        'fig_h.pdf',
        'fig_a_report.pdf',
        'fig_h_report.pdf',
        'fig_a_corr.pdf',
        'fig_h_corr.pdf',
        'fig_mat.pdf',
    ],
    label_folder_in=['', ''],
    label_folder_out='remove',
    label_file_in='',
    label_file_out='',
    graph=None,
    times=None,
    nodes=None,
    ew=None,
    a=None,
    h=None,
    epoch=10,  # How many times repead the node removal action
    remove=0.5,  # Stop if X ratio of nodes were removed (even if not reached X epoch)
    step=10,  # E.g. Remove 10 % or 10 nodes at each epoch    
    strategy_a='a',  # 'a' or 'h' = authority or hub    
    strategy_b='t',  # 't' or 's' = temporal or static
    strategy_c='r',  # 'r' or 'n' = ratio or number
    strategy_d=1,  # Score-based or random approach
    time_window=(5, 12),  # Remove high rank nodes between 5 pm - 12 am
    actions=[0, 2],
    output=True,
    plot_times=False,
    save_networks=True,
    return_graphs=True,
    return_scores=True,
):
    """
    Strategy
        A: Score
            a = authority
            h = hub
        B: Network
            s = static
            t = temporal
        C: Size
            n = number number of nodes to be removed (default = 1)
            r = ratio of nodes to be removed (default = 1 %)
        D: method
            0) randomly
            1) centrality-based (i.e. temporal HITS)
            2) high rank nodes at specific time window
            3) degree-based (in future ...)
    Actions
        0) Convert TON to TEMPORAL model and save it
        1) Convert TEMPORAL model to STATIC model
        2) Calculate temporal HITS on modified network
            A) Calculate new Edge-Weghts and save
            B) Calculate HITS
            C) Analyze HITS
        ---
        Future methods:
        2) intersection similarity
        3) centrality robustness
        4) influence maximization
        5) network diameter (90 threshhold)
        6) Number and size of gient connected components (CC)
            - Is it still one big CC (time = 1 -> T) or it is broken into pices
            - How many CC's or time windows (e.g. start-t2, t2-t10, t10-end)
        7) Average pair-wise (tempora and topological=hop) distance
        8) Network reachability
        9) Epidemic treshhold (spread of information or disease)
        10) Shanon diversity
    """
    # Edit paths
    # file_out = path_edit(file_out,folder_out,label_file_out,label_folder_out)

    # Graph
    if graph is None:
        graph = ton_bt_read(
            folder_in[0],
            [file_in[0]],
            label_folder_in[0],
            label_file_in,
        )

    # Nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in[0],
            [file_in[1]],
            label_folder_in[0],
            label_file_in,
        )
    N = len(nodes)

    # Times
    if times is None:
        times = temporal_bt_times_read(
            folder_in[0],
            [file_in[2]],
            label_folder_in[0],
            label_file_in,
        )
    times = list(times)
    T = len(times)

    # Keep times in separate object
    times_original = times.copy()

    # Edge weight
    if ew is None:
        ew = ew_read(
            folder_in[0],
            [file_in[3]],
            label_folder_in[0],
            label_file_in,
        )

    # # Read HITS
    if a is None or h is None:
        a, h = hits_read(
            folder_in[1],
            file_in[4:6],
            label_folder_in[1],
            label_file_in,
        )

    # Number of nodes and edges of TON graph
    N_ton = graph.number_of_nodes()
    M_ton = graph.number_of_edges()

    # Size of nodes and timestamp of temporal network after node removal
    # Will be updated later but better to have it as global variable
    N_tem = 0
    T_tem = 0
    M_tem = 0

    # Print network info before any node removal
    if output: print(nx.info(graph), '\n')

    # Create list of scores by sorting the key of dictionary
    a_values, h_values = [], []
    for k in sorted(a.keys(), reverse=False):  # Ascending
        # Score of node 1 to the last node id number
        a_values.append(a[k])
        h_values.append(h[k])

    # Create A and H matrices by iterating through sorted list of node
    A = np.reshape(a_values, (T + 1, N)).T
    H = np.reshape(h_values, (T + 1, N)).T

    # Create average score of nodes and timestamps
    A_avg_node = np.mean(A, axis=1)
    A_avg_time = np.mean(A, axis=0)
    H_avg_node = np.mean(H, axis=1)
    H_avg_time = np.mean(H, axis=0)

    # Time Analysis
    t_first = times[0].strftime('%d %B\n%I %p')
    t_last = times[-1].strftime('%d %B\n%I %p')
    t_0 = times[0] - pd.Timedelta(1, unit='h')

    # Distribution of timestamps over days and weeks of the month
    # Also finding time index of timestamp that hour change to 7am, 12p, 5pm, and 12am
    t_day_idx = []
    t_day_week = []
    t_day_date = []  # 7 am (or earliest time of the day)
    t_hour_12 = []
    t_hour_17 = []
    # Save index of each timestamp belong to what time windows of the day
    # [ [7 am - noon], [12 pm - 5 pm], [5 pm - 12 am] ]
    for i, time in enumerate(times):
        if t_0.weekday() != time.weekday():  # When day of the week changes
            t_day_idx.append(
                i
            )  # Index of the first hour of each day (expected to be 7 am)
            t_day_week.append(time.strftime('%a'))  # Monday ...
            t_day_date.append(time.strftime('%d'))  # 1st, 2nd, ..., 31
            t_0 = time
        if time.strftime('%H') == '12':
            t_hour_12.append(i)
        if time.strftime('%H') == '17':
            t_hour_17.append(i)

    # Adding first timestamp same as timestamp 1
    # Just because of temporal network model requires this method
    times.insert(0, times[0])  # Make times list to size T + 1
    # NOTE Times list and its size is changed here ...

    # Best hour and day in terms of highest authority average score
    days_a = defaultdict(list)
    hours_a = defaultdict(list)
    for i, score in enumerate(A_avg_time):
        days_a[times[i].weekday()].append(score)
        hours_a[times[i].hour].append(score)
    days_a_sum = {k: sum(v) / len(v) for k, v in days_a.items()}
    hours_a_sum = {k: sum(v) / len(v) for k, v in hours_a.items()}

    # Day-Hour matrix score
    # 7 days and 18 hours [7 am ... 11 pm] , missing 6 hours of [12 am ... 6 am]
    day_hour_a = np.zeros((7, 18))
    day_hour_a_count = np.zeros((7, 18))
    for i, t in enumerate(times):
        day_hour_a[t.dayofweek][t.hour - 6] += A_avg_time[i]
        day_hour_a_count[t.dayofweek][t.hour - 6] += 1
    # To avoid ZeroDivisionError, replace all 0 with 1
    # Still average produce 0 because observe/count (0/1=0)
    day_hour_a_count_new = day_hour_a_count.copy()
    day_hour_a_count_new[day_hour_a_count == 0] = 1
    day_hour_a_avg = np.divide(day_hour_a, day_hour_a_count_new)[:, 1:]

    # Best hour and day in terms of highest hub average score
    days_h = defaultdict(list)
    hours_h = defaultdict(list)
    for i, score in enumerate(H_avg_time):
        days_h[times[i].weekday()].append(score)
        hours_h[times[i].hour].append(score)
    days_h_sum = {k: sum(v) / len(v) for k, v in days_h.items()}
    hours_h_sum = {k: sum(v) / len(v) for k, v in hours_h.items()}

    # Day-Hour matrix score
    day_hour_h = np.zeros((7, 18))
    day_hour_h_count = np.zeros((7, 18))
    for i, t in enumerate(times):
        day_hour_h[t.dayofweek][t.hour - 6] += H_avg_time[i]
        day_hour_h_count[t.dayofweek][t.hour - 6] += 1

    day_hour_h_count_new = day_hour_h_count.copy()
    day_hour_h_count_new[day_hour_h_count == 0] = 1
    day_hour_h_avg = np.divide(day_hour_h, day_hour_h_count_new)[:, 1:]

    if plot_times:
        ax = plt.axes()
        sns.heatmap(
            day_hour_a_count[:, 1:],
            linewidth=0.5,
            cmap='YlGnBu',
            xticklabels=list(range(7, 24)),
            yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            ax=ax
        )
        ax.set_title('Day and hour frequency of timestamps')
        plt.show()
        # ---
        ax = plt.axes()
        sns.heatmap(
            day_hour_a_avg,
            linewidth=0.5,
            cmap='YlGnBu',
            xticklabels=list(range(7, 24)),
            yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        )
        ax.set_title('Average authority score over days and hours')
        plt.show()
        # ---
        ax = plt.axes()
        sns.heatmap(
            day_hour_h_avg,
            linewidth=0.5,
            cmap='YlGnBu',
            xticklabels=list(range(7, 24)),
            yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        )
        ax.set_title('Average hub score over days and hours')
        plt.show()

    epoch_0 = 0  # How many times the algorimth is looped
    remove_0 = 0  # ratio of removed nodes

    # Number of nodes or ratio
    if strategy_c == 'n':
        # Step should be an integer number
        # print(f'Removing {step} nodes or equevelant of {step/N_ton*100:.2f} % of all the nodes at every epoch\n')
        pass  # Do nothing, if not printing
    elif strategy_c == 'r':
        # Step should be a float ratio [0.0  ... 1.0]
        # print(f'Removing of {step*100:.2f} % of all the nodes or equevelant of {int(N_ton*step)} nodes at every epoch\n')
        # Edit number of removing nodes based on input ratio
        step = int(N_ton * step)

    # Iterative over chunk of X nodes with high HITS scores
    bd = []
    tops = []
    if strategy_a == 'a':
        bd = breakdown(a_values, step)
        tops = iter(bd)
    elif strategy_a == 'h':
        bd = breakdown(h_values, step)
        tops = iter(bd)

    # If wants to return the calculated graphs at the end
    temporals = {}  # Dict of temporal graphs
    tons = {}  # Dict of TON model graphs
    auts = {}  # Dict of authority scores
    hubs = {}  # Dict of hub scores
    taus = {}  # Dict of kendall tau correlations
    isims = {}  # Dict if intersection similarities

    # Add initial graphs and scores here
    temporal = temporal_bt_read()
    M = temporal.number_of_edges()

    if return_graphs:
        tons[epoch_0] = graph
        temporals[epoch_0] = temporal
    if return_scores:
        auts[epoch_0] = a
        hubs[epoch_0] = h

    epoch_0 += 1  # 1
    selected_nodes = []
    removed_total = []
    # Keep removing nodes until one of the conditions happen
    # while epoch_0 <= epoch and remove_0 <= remove:
    for _ in range(1, len(bd)):
        if epoch_0 > epoch or remove_0 >= remove:
            break
        else:
            if output: print(f'EPOCH {epoch_0}:\n---------')

        # Create saving folders and file names
        label_epoch = label_folder_out + '/' + str(epoch_0)
        file_out_new = path_edit(
            file_out,
            folder_out,
            label_file_out,
            label_epoch,
        )
        # Create folders (if don't exist)
        os.makedirs(os.path.join(folder_out, label_epoch), exist_ok=True)

        # Empty list of node to be removed ...
        selected_nodes.clear()

        # Method 0
        if strategy_d == 0:
            if output:
                print(
                    'Removing', step, 'nodes out of', graph.number_of_nodes(),
                    'selected randomly ...\n'
                )

            # Pick a set of nodes uniformy random
            # np.random.seed(0)
            selected_nodes = list(
                np.random.choice(graph.nodes, size=step, replace=False)
            )
            removed_total.extend(selected_nodes)
            np.savetxt(
                file_out_new[0],
                selected_nodes,
                delimiter=',',
                fmt='%s',
            )
            # if output: print(selected_nodes)

        # Method 1
        elif strategy_d == 1:
            if output:
                print(
                    'Removing', step, 'nodes out of', graph.number_of_nodes(),
                    'selected based on temporal HITS scores ...\n'
                )

            # Select top ranked HITS score nodes
            selected_nodes = next(tops)
            removed_total.extend(selected_nodes)
            np.savetxt(
                file_out_new[0],
                selected_nodes,
                delimiter=',',
                fmt='%s',
            )
            # if output: print(selected_nodes)

        # Method 3 TODO
        elif strategy_d == 2:
            if output:
                print(
                    'Removing', step, 'nodes out of', graph.number_of_nodes(),
                    'selected based on average HITS scores and in time window of ',
                    time_window, ' ...\n'
                )

            # Sort best node with highest average score
            ids_a = A_avg_node.argsort()[::-1]
            ids_h = H_avg_node.argsort()[::-1]
            # Select based on desired ratio
            selected_a = ids_a[epoch * step:(epoch + 1) * step]
            selected_h = ids_h[epoch * step:(epoch + 1) * step]
            # Filter for desired time window
            selected_noodes = []
            selected_times = list(range(T + 1))
            for n in selected_a:
                selected_noodes.extend([N * t + n for t in selected_times])
            removed_total.extend(selected_nodes)

        # Remove selected nodes
        graph.remove_nodes_from(selected_nodes)
        graph.name = 'Time-ordered Network (' + str(epoch_0) + ')'

        # If returning
        if return_graphs: tons[epoch_0] = graph

        # Save
        if save_networks: nx.write_gpickle(graph, file_out_new[1])

        # Print network statistics after node removal
        if output:
            print(nx.info(graph))
            print(
                f'Removed {N_ton - graph.number_of_nodes()} nodes out of {N_ton}'
                f' or {(N_ton - graph.number_of_nodes()) / N_ton * 100:.2f} % and {M_ton - graph.number_of_edges()}'
                f' edges out of {M_ton} or {(M_ton - graph.number_of_edges()) / M_ton * 100:.2f} %'
            )
            con = nx.algorithms.components.is_weakly_connected(graph)
            den = nx.classes.function.density(graph)
            # dim = graph.number_of_nodes()
            cc = 1
            if not con:
                cc = nx.algorithms.components.number_weakly_connected_components(
                    graph
                )
                largest_cc = max(
                    nx.weakly_connected_components(graph), key=len
                )
                # dim = nx.algorithms.distance_measures.diameter(largest_cc)
            # else:
            # dim = nx.algorithms.distance_measures.diameter(graph)
            print(f'Network is ', end='')
            if not con: print('not ', end='')
            print(
                f'connected with density of {den} and {cc} connected components.'
            )
            print()

        # ACTIONS
        # -------

        # Convert TON to TEMPORAL then save
        if 0 in actions:
            temporal = ton_bt_to_temporal(
                temporal=graph,
                label_folder_out=label_epoch,
                label_graph=str(epoch_0)
            )
            # Update number of nodes and timestamp in temporal graph (not TON)
            N_tem = file_line_count(file_out_new[3])
            T_tem = file_line_count(file_out_new[4])
            M_tem = temporal.number_of_edges()
            if output:
                print(
                    f'Removed {N - N_tem} nodes out of {N}'
                    f' or {(N - N_tem) / N*100:.2f} % and {M - M_tem}'
                    f' edges out of {M} or {(M - M_tem) / M * 100:.2f} %'
                    f' and have {T_tem} timestamps out of {T}.'
                )
                print()
            # If returning
            # Save original temporal graph as well
            if return_graphs: temporals[epoch_0] = temporal

        # Convert TEMPORAL to STATIC then save
        if 1 in actions:
            pass

        # Calculate HITS on new TON graph
        if 2 in actions:
            # Edge-Weight
            ew = edge_weight(
                label_folder_in=label_epoch,
                label_folder_out=label_epoch,
                graph=graph,  # updated
                nodes=nodes,  # original
                times=times_original,  # original
                version=3,
                omega=1,
                epsilon=1,
                gamma=0.0001,
                distance=0.1,
                alpha=0.5,
            )
            # HITS
            a_new, h_new = hits(
                folder_out=NETWORK,
                label_folder_in=label_epoch,
                label_folder_out=label_epoch,
                graph=graph,  # updated
                nodes=nodes,  # original
                times=times_original,  # original
                ew=ew,  # updated
                version=3,
                sigma=0.85,
                max_iter=100,
            )
            auts[epoch_0] = a_new
            hubs[epoch_0] = h_new
            # Conditional HITS scores
            hits_conditional(
                folder_in=[NETWORK, NETWORK],
                folder_out=NETWORK,
                label_folder_in=label_epoch,
                label_folder_out=label_epoch,
                a=a_new,
                h=h_new,
                N=N,
                T=T,
                removed=True,
            )
            # Analyze HITS
            hits_analyze(
                folder_in=[NETWORK, NETWORK],
                folder_out=NETWORK,
                label_folder_in=[label_epoch, label_epoch],
                label_folder_out=label_epoch,
                graph=graph,  # updated
                nodes=nodes,  # original
                times=times_original,  # original
                ew=ew,  # updated
                a=a_new,  # updated
                h=h_new,  # updated
                top=2,
                section=4,
                report_num=-1
            )

        # Update epoch and ...
        epoch_0 += 1
        remove_0 = (N_ton - graph.number_of_nodes()) / N_ton

    # Kendall tau for HITS score after removal - implementation # 1
    if 3 in actions:
        for e1 in auts:
            for e2 in auts:
                if e1 != e2 and (e1, e2) not in taus and (e2, e1) not in taus:
                    # Find common number of nodes or key
                    comm = set(auts[e1]).intersection(set(auts[e2]))
                    scores1 = {}
                    scores2 = {}
                    for key in comm:
                        scores1[key] = auts[e1][key]
                        scores2[key] = auts[e2][key]
                    rank1 = rank(scores1, return_rank=True).sort_index().values
                    rank2 = rank(scores2, return_rank=True).sort_index().values
                    tau, pvalue = stats.kendalltau(rank1, rank2)
                    taus[(e1, e2)] = tau
                    taus[(e2, e1)] = tau
        # Covert tau dict to 2d array for visualization
        taus_array = np.eye(epoch_0 + 1)
        for key in taus:
            taus_array[key[0], key[1]] = taus[key]
        # Plot the Kendalls tau
        fig, ax = plt.subplots(figsize=(16, 12))
        ax = sns.heatmap(
            taus_array,
            linewidth=0.5,
            annot=True,
            cmap='YlGnBu',
            ax=ax,
        )
        ax.invert_yaxis()
        plt.title(
            'Kendall Tau of Nodes Rank from HITS\nAfter Each Node-Removal Iteration'
        )
        plt.tight_layout()
        # plt.show()
        plt.savefig('kendall_tau_removal1.pdf', dpi=400)

    # Kendall tau for HITS score after removal - implementation # 2
    if 4 in actions:
        n_set = set(np.arange(N_ton))  # [0 ... 8428)
        for e1 in auts:
            for e2 in auts:
                if e1 != e2 and (e1, e2) not in taus and (e2, e1) not in taus:
                    # Find missing/removed nodes from new HITS score
                    # Add score of 0 for removed nodes
                    temp1 = auts[e1].copy()
                    temp2 = auts[e2].copy()
                    rem = n_set.difference(set(temp1))
                    for key in rem:
                        temp1[key] = 0
                    rem = n_set.difference(set(temp2))
                    for key in rem:
                        temp2[key] = 0
                    # Now we get MAX rank of 8427 for removed nodes in both score sets
                    rank1 = rank(temp1, method='max',
                                 return_rank=True).sort_index().values
                    rank2 = rank(temp2, method='max',
                                 return_rank=True).sort_index().values
                    tau, pvalue = stats.kendalltau(rank1, rank2)
                    taus[(e1, e2)] = tau
                    taus[(e2, e1)] = tau
        # Covert tau dict to 2d array for visualization
        taus_array = np.eye(epoch_0)
        for key in taus:
            taus_array[key[0], key[1]] = taus[key]
        # Plot the Kendalls tau
        fig, ax = plt.subplots(figsize=(16, 12))
        ax = sns.heatmap(
            taus_array,
            linewidth=0.5,
            annot=True,
            cmap='YlGnBu',
            ax=ax,
        )
        ax.invert_yaxis()
        plt.title(
            'Kendall Tau of Nodes Rank from HITS\nAfter Each Node-Removal Iteration'
        )
        plt.tight_layout()
        # plt.show()
        plt.savefig('kendall_tau_removal2.pdf', dpi=400)

    # Intersection similarity
    if 5 in actions:
        for e1 in auts:
            for e2 in auts:
                if e1 != e2 and (e1,
                                 e2) not in isims and (e2, e1) not in isims:
                    isims[(e1, e2)
                          ] = isim_hits(auts[e1], auts[e2], str(e1), str(e2))
                    isims[(e2, e1)] = 0

    return {
        'temporal': temporals,
        'ton': tons,
        'hits': (auts, hubs),
        'tau': taus,
        'isim': isims,
    }


# ------------
# Reachability
# ------------

# def reachability(
#     folder_in=NETWORK,
#     folder_out=NETWORK,
#     file_in=[
#         'bt_ton_network.gpickle',
#         'bt_temporal_nodes.csv',
#         'bt_temporal_times.csv',
#         'bt_ton_weights.csv',
#     ],
#     file_out=[
#         'reachability.csv',
#     ],
#     label_folder_in='',
#     label_folder_out='',
#     label_file_in='',
#     label_file_out='',
#     graph=None,
#     times=None,
#     nodes=None,
#     ew=None,
#     # output=True,
#     # plot=False,
#     save=True,
# ):
#     # Edit paths
#     file_out = path_edit(
#         file_out,
#         folder_out,
#         label_file_out,
#         label_folder_out,
#     )

#     # Reading graph
#     if graph is None:
#         graph = ton_bt_read(
#             folder_in,
#             [file_in[0]],
#             label_folder_in,
#             label_file_in,
#         )

#     # Reading nodes
#     if nodes is None:
#         nodes = temporal_bt_nodes_read(
#             folder_in,
#             [file_in[1]],
#             label_folder_in,
#             label_file_in,
#         )
#     N = len(nodes)

#     # Reading times
#     if times is None:
#         times = temporal_bt_times_read(
#             folder_in,
#             [file_in[2]],
#             label_folder_in,
#             label_file_in,
#         )
#     T = len(times)

#     # Reading edge weight
#     # if ew is None:
#     #     ew = ew_read(
#     #         folder_in,
#     #         [file_in[3]],
#     #         label_folder_in,
#     #         label_file_in,
#     #     )

#     # All possible paths
#     all_paths = {}
#     for n1 in range(N):
#         for t1 in range(T):
#             for n2 in range(N):
#                 for t2 in range(T):
#                     if n1 != n2 and t1 < t2:
#                         all_paths[(n1, n2, t1, t2)] = np.inf

#     # All existing network shortest path lenght (SPL)
#     spl = dict(nx.all_pairs_shortest_path_length(graph))

#     # Convert N1 -> N2 to Path of (Parent N1, Parent N2, t1, t2)
#     # Save all reachable paths P(n1,n2,t1,t2) in a dataframe
#     df = []
#     tdistances = []
#     reached = set()
#     for n1 in spl:
#         # Parent of source node
#         p1 = n1 % N
#         t1 = n1 // N
#         for n2 in spl[n1]:
#             p2 = n2 % N
#             t2 = n2 // N
#             # Nodes with different parents and timestamps
#             if p1 != p2:
#                 # Add path to reachability dataframe
#                 df.append((p1, p2, t1, t2))
#                 tdistances.append(spl[n1][n2])
#                 # Update paths dictionary
#                 all_paths[(p1, p2, t1, t2)] = spl[n1][n2]
#                 # Saved observed paths so can remaining can be calculated
#                 reached.add((p1, p2, t1, t2))
#     # Add unexisting paths to dataframe
#     rem = set(all_paths.keys()).difference(reached)
#     for path in rem:
#         df.append((path[0], path[1], path[2], path[3]))
#         tdistances.append(np.inf)

#     # Create dataframe of reachability
#     # df = pd.DataFrame(df, columns=['n1', 'n2', 't1', 't2', 'd'])
#     df = pd.DataFrame(df, columns=['n1', 'n2', 't1', 't2'])
#     df['d'] = tdistances

#     # Check paths for hop number or node-distance (not temporal distance)
#     ndistances = []
#     between = {}
#     # TODO fix following
#     for n1 in spl:
#         p1 = n1 % N
#         t1 = n1 // N
#         for n2 in spl[n1]:
#             p2 = n2 % N
#             t2 = n2 // N
#             if p1 != p2:
#                 q = df.loc[(df['n1'] == p1) & (df['n2'] == p2) &
#                            (df['t1'] >= t1) & (df['t2'] <= t2)]
#                 # If there is only one node (1-hop) between two nodes A -> B
#                 if q.d.min() == 1:
#                     ndistances.append(1)

#                 # If it is not a direct connection and intermediate nodes are incolved
#                 if q.d.min() > 1:
#                     # Look at the different shortest paths
#                     sps = [np.array(p)%N for p in nx.all_shortest_paths(T, source=n1, target=n2)]
#                     # Extract the intermediate nodes
#                     # Save in a dictionary based on length
#                     spd = defaultdict(list)
#                     for x in sps:
#                         # Remove consecutive nodes (same node, next or previous time)
#                         temp = [v for i, v in enumerate(x) if i == 0 or v != x[i-1]]
#                         spd[len(temp)].append(temp)
#                     # Check the shortest extracted paths
#                     min_hop = min(spd.keys())
#                     ndistances.append(min_hop)
#                     for x in spd[min_hop]:
#                         int_nodes = np.unique(x[1:-1])
#                     sps = spd[min(spd.keys())]

#     # Sort the dataframe
#     df.sort_values(
#         by=['n1', 'n2', 't1', 't2'],
#         inplace=True,
#         ignore_index=True,
#     )

#     # Save paths distance dataframe
#     df.to_csv(file_out[0], header=False, index=False)


def reachability(
    folder_in=NETWORK,
    folder_out=NETWORK,
    file_in=[
        'bt_ton_network.gpickle',
        'bt_temporal_nodes.csv',
        'bt_temporal_times.csv',
        'bt_ton_weights.csv',
    ],
    file_out=[
        'reachability.csv',
        'reachability.p',
    ],
    label_folder_in='',
    label_folder_out='',
    label_file_in='',
    label_file_out='',
    graph=None,
    times=None,
    nodes=None,
    ew=None,
    output=True,
    save=True,
):
    # Edit paths
    file_out = path_edit(
        file_out,
        folder_out,
        label_file_out,
        label_folder_out,
    )

    # Reading graph
    if graph is None:
        graph = ton_bt_read(
            folder_in,
            [file_in[0]],
            label_folder_in,
            label_file_in,
        )

    # Reading nodes
    if nodes is None:
        nodes = temporal_bt_nodes_read(
            folder_in,
            [file_in[1]],
            label_folder_in,
            label_file_in,
        )
    N = len(nodes)

    # Reading times
    if times is None:
        times = temporal_bt_times_read(
            folder_in,
            [file_in[2]],
            label_folder_in,
            label_file_in,
        )
    T = len(times)

    # Reading edge weight
    # if ew is None:
    #     ew = ew_read(
    #         folder_in,
    #         [file_in[3]],
    #         label_folder_in,
    #         label_file_in,
    #     )

    # All possible paths
    all_paths = {}
    for n1 in range(N):
        for t1 in range(T + 1):
            for n2 in range(N):
                for t2 in range(T + 1):
                    if n1 != n2 and t1 < t2:
                        all_paths[(n1, n2, t1, t2)] = {'d': np.inf}

    # All existing network shortest path lenght (SPL)
    spl = dict(nx.all_pairs_shortest_path_length(graph))

    # Convert N1 -> N2 to Path of (Parent N1, Parent N2, t1, t2)
    # Save all reachable paths P(n1,n2,t1,t2) in a dataframe
    reached = []
    r = 0

    for n1 in spl:
        # Parent of source node
        p1 = n1 % N
        t1 = n1 // N
        for n2 in spl[n1]:
            p2 = n2 % N
            t2 = n2 // N
            # Nodes with different parents and timestamps
            if p1 != p2:
                # Add path lenght
                reached.append((p1, p2, t1, t2, spl[n1][n2]))
                # Add temporal distance to path
                all_paths[(p1, p2, t1, t2)]['d'] = spl[n1][n2]
                # Add the number of reached paths
                r += 1

    if output:
        print(f'Number of reachable paths is {r}', end='')
        print(f'from a total of {len(all_paths)} possoble paths or ', end='')
        print(f'{r/len(all_paths)*100:0.2f} %')

    # Create dataframe of reachability -> Easy to use for path filtering
    df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])

    # Check paths for hop-number or node-distance
    cc = 0  # Counter
    cr = 0  # Counter % passed
    cl = int(0.01 * r)  # Counter limit = 1 % of all paths
    for (p1, p2, t1, t2), val in all_paths.items():
        d = val['d']
        if d < np.inf:
            cc += 1
            if cc % cl == 0:
                cr += 1
                print(f'Processed {cr} % of all data ...')
            mind = 0
            # Check if there is a path with shorter temporal distance
            q = df.loc[(df['n1'] == p1) & (df['n2'] == p2) & (df['t1'] >= t1) &
                       (df['t2'] <= t2) & (df['d'] < d)]
            # Check if it is direct edge or 1-hop between two nodes
            # But in a shoeter time interval included in larger time window
            if q.d.min() == 1:
                # ndistances.append(1)
                mind = 1
                all_paths[(p1, p2, t1, t2)]['h'] = 1

            # If it is not a direct connection
            # And there are some intermediate nodes incolved
            if q.d.min() > 1:
                # Look at all shortest paths in that time range
                sps = [
                    np.array(p) % N for p in nx.all_shortest_paths(
                        graph,
                        source=(N * t1) + p1,
                        target=(N * t2) + p2,
                    )
                ]
                # Extract  intermediate nodes
                # And save in a dictionary based on path length
                spd = defaultdict(list)
                for x in sps:
                    # Remove consecutive nodes (same node, next or previous time)
                    temp = [
                        v for i, v in enumerate(x) if i == 0 or v != x[i - 1]
                    ]
                    spd[len(temp)].append(temp)
                # Check only the shortest extracted paths
                mind = min(spd.keys())
                spd = spd[mind]
                # Minus 1 (-1) because hop length is 1 less than number of nodes
                all_paths[(p1, p2, t1, t2)]['h'] = mind - 1
                # Unique intermediate nodes invole in shortest path
                paths = paths = set()
                for x in spd:
                    paths.add(tuple(x[1:-1]))
                all_paths[(p1, p2, t1, t2)]['p'] = [list(x) for x in paths]
        else:
            # Path does not exist
            # Distance is infinity
            # Then add to REACHED list to create new DF
            reached.append((p1, p2, t1, t2, np.inf))

    # Create reachability again LOL
    df = pd.DataFrame(reached, columns=['n1', 'n2', 't1', 't2', 'd'])
    # Then sort ...
    df.sort_values(
        by=['n1', 'n2', 't1', 't2'],
        inplace=True,
        ignore_index=True,
    )
    # Finally save reachability dataframe and Hop dictionary
    if save:
        df.to_csv(file_out[0], header=False, index=False)
        with open(file_out[1], 'wb') as fp:
            pickle.dump(all_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)
