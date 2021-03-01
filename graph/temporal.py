def temporal_bt_create(
    file_input=['db/bt.csv', 'db/uiuc.db'],
    file_output=[
        'network/bt_temporal_network.gpickle',
        'network/bt_bipartite_network.gpickle',
        'network/bt_temporal_edgelist.csv',
        'network/bt_bipartite_edgelist.csv', 'network/bt_temporal_times.csv',
        'network/bt_bipartite_times.csv', 'network/bt_temporal_nodes.csv',
        'network/bt_bipartite_nodes.csv', 'network/bt_temporal_weights.csv',
        'network/bt_bipartite_weights.csv',
        'network/bt_temporal_weights_scaled.csv',
        'network/bt_bipartite_weights_scaled.csv'
    ],
    input_label='',
    output_label='',
    output_times=False,
    output_network=True,
    save_times=True,
    save_nodes=True,
    save_weights=True,
    save_network_db=True,
    save_network_csv=True,
    save_network_file=True):
    """
    Read CSV dataset with 3 columns of: node-1, node-2, timestamp
    Then create an (aggregated) temporal network using Multiedge Directed Graph
    """

    # Modify input & output file names
    file_input = label_amend(file_input, input_label)
    file_output = label_amend(file_output, output_label)

    # Create empty network
    graph = nx.MultiDiGraph()  # G: user-user interaction network
    bipartite = nx.MultiDiGraph()  # B: (28)user-others biparite network

    # Read the dataset
    print('Reading dataset ... ')
    data = pd.read_csv(file_input[0],
                       header=None,
                       names=['user', 'mac', 'time'],
                       parse_dates=['time'])
    print('Done!\n')

    # Print Time : Frequency
    if output_times:
        times = pd.Series(sorted(data.time.unique()))
        print(len(times), 'timestamps')
        print('Timestamp : Frequency')
        for t_size in data.groupby('time').size().iteritems():
            print('{}) {} : {}'.format(times[times == t_size[0]].index[0],
                                       t_size[0], t_size[1]))
        print()

    # Create timestamp list
    times = []
    times_bipartite = []

    # Dictionary {time:{(user-1,user-2):weight}}
    time_edges = defaultdict()  # User -> User
    time_bipartite_edges = defaultdict()  # Users -> other devices

    # Group interactions by time
    for key_time, group_user_mac in data[['user',
                                          'mac']].groupby(data['time']):
        # print()
        # print('Time:\n', key_time)
        # print('Group:\n', group_user_mac)

        # Normal (co-location) graph edges in filtered timestamp
        temp_edges = []
        for key_mac, group_connection in group_user_mac.groupby(['mac'
                                                                 ])['user']:
            # print('Mac:', key_mac)
            # print('Group:', group_connection)

            # Users of connecting to filtered MAC
            temp_users = list(group_connection.unique())
            # print('Unique users:', temp_users)

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
        # for key_mac, group_connection in group_user_mac.groupby(['mac'])['user']:
        for key_mac, group_connection in group_user_mac.groupby(
            ['mac', 'user']):
            # User connected to filtered MAC with X number of times
            # print('{}\t{}'.format(key_mac, len(group_connection)))
            bipartite_edges[key_mac] = len(group_connection)

        # Add edges of this time (with their strength) to dictionary
        time_bipartite_edges[key_time] = bipartite_edges

    # Normal (co-location) network
    l1, l2, l3, l4 = [], [], [], []  # time, node, node, weight
    for k1, v1 in time_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_graph = pd.DataFrame(list(zip(l1, l2, l3, l4)),
                              columns=['t', 'u', 'v', 'w'])

    # Weights
    # Scale the weights of connection [0-1]
    X = [[entry] for entry in data_graph['w']]
    if save_weights: np.savetxt(file_output[8], X, delimiter=',', fmt='%s')

    # Plot the distribution of original weights
    plt.figure()
    # ax = sns.distplot(X, bins=10)
    ax = sns.distplot(X,
                      bins=max(X),
                      kde=True,
                      hist_kws={
                          "linewidth": 15,
                          'alpha': 1
                      })
    ax.set(xlabel='Distribution', ylabel='Frequency')

    # Max-Min Normalizer (produce many zeros)
    # transformer = MinMaxScaler()
    # X_scaled = transformer.fit_transform(X)
    # Returning column to row vector again
    # X_scaled = [entry[0] for entry in X_scaled]

    # Quantile normalizer (normal distribution)
    # transformer = QuantileTransformer()
    # Quantile normalizer (uniform distribution)
    transformer = QuantileTransformer(n_quantiles=1000,
                                      output_distribution='uniform')
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
    plt.figure()
    # ax = sns.distplot(X_scaled, bins=10)
    ax = sns.distplot(X_scaled, bins=10, kde=True, hist_kws={'alpha': 1})
    ax.set(xlabel='Distribution ', ylabel='Frequency')

    # Save scaled weights back to DF
    data_graph['w'] = X_scaled

    # Save the new weights
    if save_weights:
        np.savetxt(file_output[10], X_scaled, delimiter=',', fmt='%s')

    # Save to DB
    if save_network_db:
        data_graph[['u', 'v', 't', 'w']].to_sql(name='bluetooth_edgelist',
                                                con=db_connect(file_input[1]),
                                                if_exists='replace',
                                                index_label='id')

    # Save to file
    if save_network_csv:
        data_graph[['u', 'v', 't', 'w']].to_csv(file_output[2],
                                                header=False,
                                                index=False)

    # Add edges to network object
    # Complete network object
    for row in data_graph.itertuples(index=True, name='Pandas'):
        graph.add_edge(getattr(row, 'u'),
                       getattr(row, 'v'),
                       t=getattr(row, 't'),
                       w=getattr(row, 'w'))

    # Save graph
    if save_network_file:
        nx.write_gpickle(graph, file_output[0])

    # Save timestamps
    if save_times:
        np.savetxt(file_output[4] + '.csv', times, delimiter=',', fmt='%s')
        # pd.DataFrame(sorted(list(times))).to_csv(file_output[4], header=None, index=False)
        times_set = set()
        for u, v, w in graph.edges(data=True):
            times_set.add(w['t'])
        times = pd.Series(sorted(list(times_set)))
        np.savetxt(file_output[4], times, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the graph
        nodes = pd.Series(sorted(list(graph.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_output[6], nodes, delimiter=',', fmt='%s')
        # pd.DataFrame(sorted(list(times))).to_csv(file_output[6], header=None, index=False)

    # Bipartite network
    # Complete network object
    l1, l2, l3, l4 = [], [], [], []
    for k1, v1 in time_bipartite_edges.items():
        for k2, v2 in v1.items():  # k2 = edge = (u,v)
            if k2[0] != k2[1]:
                l1.append(k1)
                l2.append(k2[0])
                l3.append(k2[1])
                l4.append(v2)
    data_bi_graph = pd.DataFrame(list(zip(l1, l2, l3, l4)),
                                 columns=['t', 'u', 'v', 'w'])

    # Weights
    X = [[entry] for entry in data_bi_graph['w']]
    if save_weights: np.savetxt(file_output[9], X, delimiter=',', fmt='%s')
    transformer = QuantileTransformer(n_quantiles=100,
                                      output_distribution='uniform')
    X_scaled = transformer.fit_transform(X)
    X_scaled = [entry[0] for entry in X_scaled]
    if save_weights:
        np.savetxt(file_output[11], X_scaled, delimiter=',', fmt='%s')
    # data_bi_graph['w'] = X_scaled

    # Save to DB
    if save_network_db:
        data_bi_graph[['u', 'v', 't',
                       'w']].to_sql(name='bluetooth_bipartite_edgelist',
                                    con=db_connect(file_input[1]),
                                    if_exists='replace',
                                    index_label='id')

    # Save to file
    if save_network_csv:
        data_bi_graph[['u', 'v', 't', 'w']].to_csv(file_output[3],
                                                   header=False,
                                                   index=False)

    # Add nodes and then edges
    # We need to add a prefix "u_" for users & "b_" for BT devices to the node id
    # So that we can distinguish them from each others
    for row in data_bi_graph.itertuples(index=True, name='Pandas'):
        # In bluetooth connections, user devices ID are repeated in all BT devices
        # So there is no need to differentiate between them, unless for some research necessary
        bipartite.add_edge(getattr(row, 'u'),
                           getattr(row, 'v'),
                           t=getattr(row, 't'),
                           w=getattr(row, 'w'))
        # Uncomment next 5 lines if wanna difrentiate users from other devices
        # node_user = 'u_' + str(getattr(row, 'u'))
        # node_ap = 'b_' + str(getattr(row, 'v'))
        # bipartite.add_node(node_user, bipartite=0)
        # bipartite.add_node(node_ap, bipartite=1)
        # bipartite.add_edge(node_user, node_ap, t=getattr(row, 't'), w=getattr(row, 'w'))

    # Save graph
    if save_network_file:
        nx.write_gpickle(bipartite, file_output[1])

    # Save timestamps
    if save_times:
        times_set = set()
        for u, v, w in bipartite.edges(data=True):
            times_set.add(w['t'])
        times_bipartite = pd.Series(sorted(list(times_set)))
        np.savetxt(file_output[5], times_bipartite, delimiter=',', fmt='%s')

    # Save nodes
    if save_nodes:
        # List of nodes in the bipartite graph
        nodes = pd.Series(sorted(list(bipartite.nodes)))
        # Save node list in a file "node.csv"
        np.savetxt(file_output[7], nodes, delimiter=',', fmt='%s')
        # pd.DataFrame(sorted(list(times))).to_csv(file_input[2], header=None, index=False)

    # Print network statistics
    if output_network:
        print('Temporal netwrok:')
        print('N =', graph.number_of_nodes())
        print('L =', graph.number_of_edges())
        print('T =', len(times))
        print('---')
        print('Bipartite Temporal netwrok:')
        print('N =', bipartite.number_of_nodes())
        print('L =', bipartite.number_of_edges())
        print('T =', len(times_bipartite))

    return graph
    # return graph, bipartite