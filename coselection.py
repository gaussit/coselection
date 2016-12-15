import bz2
import itertools
import pandas as pd
from igraph import *
from scipy.sparse import triu
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize


def node_extractor(dataframe, *columns):
    """
    Extracts the set of nodes from a given dataframe.
    :param dataframe: dataframe from which to extract the node list
    :param columns: list of column names that contain nodes
    :return: list of all unique nodes that appear in the provided dataset
    """
    data_list = [dataframe[column].unique().tolist() for column in columns]

    return list(set(itertools.chain.from_iterable(data_list)))


def dataframe_to_matrix(dataframe, id_dictionary, drop_diagonal=False):
    """
    Takes a dataframe and converts it to a sparse matrix and computes the affiliation matrix.
    :param dataframe: dataframe which is to be converted to matrices
    :param id_dictionary: dictionary which assigns each node to a unique id
    :param drop_diagonal: drop the diagonal (self loop) if enabled
    :return: yields a matrix if sub-dataframe is not empty, otherwise yields 'nan'
    """

    for time in dataframe['tim'].unique():
        for domain in dataframe['dom'].unique():
            data_dom = dataframe[(dataframe['tim'] == time) & (dataframe['dom'] == domain)]

            if len(data_dom) != 0:
                rows = [id_dictionary[data] for data in data_dom['agn'].values]
                cols = [id_dictionary[data] for data in data_dom['fct'].values]

                # convert to matrix
                data_t = data_dom['sel'].tolist() # selections or weights
                matrix = coo_matrix((data_t, (rows, cols)))

                # matrix stuff
                matrix_c = matrix.tocsr()
                matrix_cn = normalize(matrix_c, norm='l1', axis=1)
                matrix_ct = csr_matrix.transpose(matrix_c)
                matrix_cf = matrix_ct * matrix_cn

                if drop_diagonal:
                    matrix_cf = matrix_cf.tolil()
                    matrix_cf.setdiag(values=0)

                matrix_cf = matrix_cf.tocoo()
                yield triu(matrix_cf)  # only keep upper triangle, since lower triangular is merely mirrored
            else:
                yield 'nan'


def dict_to_dataframe(dictionary):
    """
    Embeds matrices from a dict matrix representation and transforms them into dataframes.
    :param dictionary: dict matrix representation, where matrices are values and a tuple consisting of time and domain
     information is the key
    :return: yields a dataframe with nodes and time and domain information
    """

    for key in dictionary.keys():
        matrix = dictionary[key]
        ss = pd.SparseSeries.from_coo(matrix)
        sd = pd.DataFrame(ss)
        sd.index.names = ['u', 'v']
        sd.columns = ['weight']
        sd['time'] = key[0]
        sd['domain'] = key[1]
        yield sd


def dataframes_to_csv(dictionary, filename):
    """
    Takes dataframes, merges them, applies filtering and exports to csv.
    :param dictionary: calls the dict_to_dataframe() function, which requires a dictionay matrix representation
    :param filename: the name of the file that is going to be created
    :return: a compressed csv with cumulative percentage for time only and time and domain.
     For each matrix in the dictionary.
    """

    df = pd.concat(dict_to_dataframe(dictionary))
    df = df.sort_values(['time', 'weight'], ascending=False)
    df["cum_perc_time"] = 100 * df.groupby('time').weight.cumsum() / df.groupby('time').weight.transform('sum')

    df = df.sort_values(['time', 'domain', 'weight'], ascending=False)
    df["cum_perc_time_domain"] = 100 * df.groupby(['time', 'domain'])['weight'].cumsum() / df.groupby(
        ['time', 'domain']).weight.transform('sum')

    df.sort_index(inplace=True)
    df.to_csv(str(filename) + '_network.csv.bz2', sep='\t', compression='bz2')


def tuples_importer(input_file, threshold):
    """
    Creates tuple representation of input data.
    :param input_file: the edge list file e.g. created by the dataframes_to_csv() function
    :param threshold: enter a threshold, up to which cumulative edge weight per time and domain data
     is going to be imported.
    :return: yields a tuple with nodes and edge information ready to import into igraph
    """

    # with open(input_file) as f:
    with bz2.open(input_file, 'rt') as f:
        next(f)  # skip header
        for line in f:
            u, v, weight, time, domain, cum_perc_time, cum_perc_time_domain = line.split()
            cum_perc_time_domain = float(cum_perc_time_domain)

            if cum_perc_time_domain <= threshold:
                u = int(u)
                v = int(v)
                weight = float(weight)
                time = int(time)
                domain = str(domain)  # specify if string or integer or float
                cum_perc_time_domain = float(cum_perc_time_domain)

                yield u, v, weight, time, domain, cum_perc_time, cum_perc_time_domain


def graph_processor(input_file, threshold, id_dictionary, include_layout=True):
    """
    Creates a compressed .graphml file which contains statistics and layout coordinates.
    Also creates a csv containing graph statistics.
    :param input_file: calls the tuples_importer() function and therefore needs an input file
    :param threshold: calls the tuples_importer() function and therefore needs a threshold
    :param id_dictionary: an id node label linkage dictionary e.g. the one created with the 'node_extractor()' function
    :param include_layout: whether to compute the layout in igraph as well and store the x and y coordinates in the
     .graphml file as well as in the *_stats.csv file
    :return: .graphml and .csv files with computed properties
    """

    g = Graph.TupleList(tuples_importer(input_file, threshold),
                        edge_attrs=('weight', 'time', 'domain', 'cum_perc_time', 'cum_perc_time_domain'),
                        directed=False)
    g.vs['label'] = [id_dictionary[node] for node in g.vs['name']]

    # graph metrics
    g.vs['degree'] = g.degree()
    g.vs['betweenness'] = g.betweenness()
    g.vs['eigenvector_centrality'] = g.eigenvector_centrality()
    g.vs['authority'] = g.authority_score()

    # compute layout
    if include_layout:
        layout = g.layout_lgl()
        g.vs['x'], g.vs['y'] = zip(*layout)

    g.write_graphmlz(str(input_file)[:-8] + '.graphml.gz')

    # metrics csv
    stats = pd.DataFrame()
    stats['degree'] = g.vs['degree']
    stats['betwennness'] = g.vs['betweenness']
    stats['eigenvector_centrality'] = g.vs['eigenvector_centrality']
    stats['authority'] = g.vs['authority']
    stats['vertex'] = g.vs['name']
    stats['label'] = [id_dictionary[int(number)] for number in stats.vertex]

    if include_layout:
        stats['x'] = g.vs['x']
        stats['y'] = g.vs['y']

    stats.to_csv(str(input_file)[:-16] + '_stats.csv', sep='\t', encoding='utf-8')


def lcc_metrics(graph):
    """
    Ddasd
    :param graph:
    :return:
    """

    nodes = len(graph.vs())
    # weights =
    cluster = graph.clusters(mode='strong')
    g_lcc = cluster.giant()
    nodes_c = len(g_lcc.vs())
    community = g_lcc.community_multilevel()
    modularity = g_lcc.modularity(community)
    assortativity = g_lcc.assortativity_degree()
    metric = [nodes, nodes_c, modularity, assortativity]

    return metric


def threshold_metrics(input_file, thresholds):
    """
    Takes an input file and a threshold list and computes assortativity and modularity for each threshold.
    :param input_file:
    :param thresholds:
    :return:
    """

    metrics = []
    g = Graph.TupleList(tuples_importer(input_file, threshold=100), # 100% because we go more fine grained below
                        edge_attrs=('weight', 'time', 'domain', 'cum_perc_time', 'cum_perc_time_domain'),
                        directed=False)

    for threshold in thresholds:
        selection = g.es.select(weight_le=threshold)  # equivalent to <= since normal operator do not work
        g_t = g.subgraph_edges(selection)
        metric = lcc_metrics(g_t)
        metric.append('all')
        metric.append(threshold)
        metrics.append(metric)

        for domain in set(g_t.es['domain']):
            selection_dom = g_t.es.select(domain=domain)
            g_d = g_t.subgraph_edges(selection_dom)
            metric_dom = lcc_metrics(g_d)
            metric_dom.append(domain)
            metric_dom.append(threshold)
            metrics.append(metric_dom)

    headers = ['nodes', 'nodes_lcc', 'modularity_lcc', 'assortativity_lcc', 'domain', 'threshold']

    return pd.DataFrame(metrics, columns=headers)
