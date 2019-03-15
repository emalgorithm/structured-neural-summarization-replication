from graph_pb2 import FeatureNode, FeatureEdge
from data_processing.text_util import split_identifier_into_parts
from pathlib import Path
from graph_pb2 import Graph
import networkx as nx
import sys
import numpy as np

sys.setrecursionlimit(10000)


def get_dataset_from_dir(dir="../corpus/r252-corpus-features/"):
    """
    Extract methods source code, names and graphs structure.
    :param dir: directory where to look for proto files
    :return: (methods_source, methods_names, methods_graphs)
    """
    methods_source = []
    methods_names = []
    methods_graphs = []

    proto_files = list(Path(dir).rglob("*.proto"))
    print("A total of {} files have been found".format(len(proto_files)))

    for i, file in enumerate(proto_files):
        file_methods_source, file_methods_names, file_methods_graph = get_file_methods_data(
            file)
        methods_source += file_methods_source
        methods_names += file_methods_names
        methods_graphs += file_methods_graph

    return methods_source, methods_names, methods_graphs


def get_file_methods_data(file):
    """
    Extract the source code tokens, identifier names and graph for methods in a source file.
    Identifier tokens are split into subtokens. Constructors are not included in the methods.
    :param file: file
    :return: (methods_source, methods_names, methods_graph) where methods_source[i] is a list of the tokens for
    the source of ith method in the file, methods_names[i] is a list of tokens for name of the
    ith method in the file, and methods_graph[i] is the subtree of the file parse tree starting
    from the method node.
    """
    adj_list, nodes, edges = get_file_graph(file)

    with file.open('rb') as f:
        class_name = file.name.split('.')

        g = Graph()
        g.ParseFromString(f.read())
        methods_source = []
        methods_names = []
        methods_graph = []
        # class_name_node = get_class_name_node(g)

        for node in g.node:
            if node.contents == "METHOD":
                method_name_node = get_method_name_node(g, node)

                # If method name is the same as class name, then method name is constructor,
                # so discard it
                if method_name_node.contents == class_name:
                    continue

                method_edges, method_nodes, non_tokens_nodes_features = get_method_edges(node.id, adj_list, nodes)
                methods_graph.append((method_edges, non_tokens_nodes_features))
                methods_names.append(split_identifier_into_parts(method_name_node.contents))

                method_source = []

                for other_node in method_nodes.values():
                    if other_node.id == method_name_node.id:
                        # Replace method name with '_' in method source code
                        method_source.append('_')
                    elif other_node.type == FeatureNode.TOKEN or other_node.type == \
                            FeatureNode.IDENTIFIER_TOKEN:
                        method_source.append(other_node.contents)

                methods_source.append(method_source)

        return methods_source, methods_names, methods_graph


def get_file_graph(file):
    """
    Compute graph for the given file.
    """
    with file.open('rb') as f:
        g = Graph()
        g.ParseFromString(f.read())
        node_ids = [node.id for node in g.node]
        edges = [(e.sourceId, e.destinationId, e.type) for e in g.edge]

        adj_list = {node: [] for node in node_ids}
        for edge in edges:
            adj_list[edge[0]].append({'destination': edge[1], 'edge_type': edge[2]})

        nodes = {node.id: node for node in g.node}

        return adj_list, nodes, edges


def get_method_edges(method_node_id, file_adj_list, file_nodes):
    """
    Compute edges of a method graph for a method starting at the node 'method_node_id'.
    """
    method_nodes_ids = []

    get_method_nodes_rec(method_node_id, method_nodes_ids, file_adj_list)
    methods_edges = []

    for node in method_nodes_ids:
        for edge in file_adj_list[node]:
            if edge['destination'] in method_nodes_ids:
                methods_edges.append((node, edge['destination']))

    method_nodes = {node_id: node for node_id, node in file_nodes.items() if node_id in
                    method_nodes_ids}

    methods_edges, non_tokens_nodes_features = remap_edges(methods_edges, method_nodes)

    return methods_edges, method_nodes, non_tokens_nodes_features


def get_method_nodes_rec(node_id, method_nodes_ids, file_adj_list):
    """
    Utilities to recursively retrieve all edges of a method graph.
    """
    method_nodes_ids.append(node_id)

    for edge in file_adj_list[node_id]:
        if edge['edge_type'] != FeatureEdge.NEXT_TOKEN and edge['destination'] not in method_nodes_ids:
            get_method_nodes_rec(edge['destination'], method_nodes_ids, file_adj_list)


def remap_edges(edges, nodes):
    """
    Remap edges so that ids start from 0 and are consecutive.
    """
    old_id_to_new_id = {}
    i = 0
    nodes_values = sorted(nodes.values(), key=lambda node: node.id)
    new_edges = []

    # Set new ids for tokens
    for node_value in nodes_values:
        if is_token(node_value):
            old_id_to_new_id[node_value.id] = i
            i += 1

    non_tokens_nodes_features = np.zeros((len(nodes_values) - len(old_id_to_new_id), 11))
    j = i
    # Set new ids for other nodes
    for node_value in nodes_values:
        if not is_token(node_value):
            old_id_to_new_id[node_value.id] = i
            non_tokens_nodes_features[i - j][node_value.type - 1] = 1
            i += 1

    for edge in edges:
        new_edges.append((old_id_to_new_id[edge[0]], old_id_to_new_id[edge[1]]))

    return new_edges, non_tokens_nodes_features


def is_token(node_value):
    return node_value.type == FeatureNode.TOKEN or node_value.type == FeatureNode.IDENTIFIER_TOKEN


def get_method_name_node(g, method_node):
    """
    Return the node corresponding to the name of a method.
    """
    method_id = method_node.id
    method_name_node_id = 0

    for edge in g.edge:
        if edge.sourceId == method_id and edge.type == FeatureEdge.ASSOCIATED_TOKEN:
            method_name_node_id = edge.destinationId
            break

    for node in g.node:
        if node.id == method_name_node_id:
            return node


def get_class_name_node(g):
    """
    :param g: graph representing the file
    :return: the node corresponding to the class identifier token
    """
    class_node = [node for node in g.node if node.contents == "CLASS"][0]
    class_associated_nodes_ids = [edge.destinationId for edge in g.edge if edge.sourceId ==
                              class_node.id and edge.type == FeatureEdge.ASSOCIATED_TOKEN]
    class_associated_nodes = [node for node in g.node if node.id in class_associated_nodes_ids]

    return class_associated_nodes[1]


def get_nx_graph(file):
    """
    Get networkx graph corresponding to a file.
    """
    nx_graph = nx.DiGraph()
    with file.open('rb') as f:
        g = Graph()
        g.ParseFromString(f.read())

        for edge in g.edge:
            edge_type = [name for name, value in list(vars(FeatureEdge).items())[8:] if value ==
                         edge.type][0]
            nx_graph.add_edge(edge.sourceId, edge.destinationId, edge_type=edge_type)
    return nx_graph
