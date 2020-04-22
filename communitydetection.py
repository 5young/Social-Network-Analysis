def caculate_communityDetection(gen_graph, label):
    partition = community.best_partition(nx.from_numpy_matrix(gen_graph))
    num_of_partition = max(partition.values())
    keys, values = zip(*partition.items())
    pred = np.array(list(values))

    return adjusted_rand_score(label, pred), num_of_partition

def newman_partition(graph, label, n):
    c = list(greedy_modularity_communities(nx.from_numpy_matrix(graph)))
    cl = np.zeros((n), int)
    for i in range(len(c)):
        c[i] = list(c[i])
        for j in range(len(c[i])):
            cl[c[i][j]] = i
    
    return adjusted_rand_score(label, cl), len(c)

def label_propagation(graph, label, n):
    c = list(label_propagation_communities(nx.from_numpy_matrix(graph)))
    cl = np.zeros((n), int)

    for i in range(len(c)):
        c[i] = list(c[i])
        for j in range(len(c[i])):
            cl[c[i][j]] = i
            
    return adjusted_rand_score(label, cl), len(c)