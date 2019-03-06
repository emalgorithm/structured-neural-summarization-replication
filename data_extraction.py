from graph_pb2 import FeatureNode, FeatureEdge
from text_util import split_identifier_into_parts
from pathlib import Path
from graph_pb2 import Graph
import networkx as nx
import sys
import numpy as np

sys.setrecursionlimit(10000)


def get_dataset_from_dir(dir="../corpus/r252-corpus-features/"):
    methods_source = []
    methods_names = []
    methods_graphs = []

    proto_files = list(Path(dir).rglob("*.proto"))
    print("A total of {} files have been found".format(len(proto_files)))

    # proto_files = [Path("../features-javac-master/Test.java.proto")]

    for i, file in enumerate(proto_files):
        # nx_graph = get_nx_graph(file)
        # if i % 100 == 0:
        print("Extracting data from file {}".format(i+1))
        file_methods_source, file_methods_names, file_methods_graph = get_file_methods_data(
            file)
        methods_source += file_methods_source
        methods_names += file_methods_names
        methods_graphs += file_methods_graph

    return methods_source, methods_names, methods_graphs


def get_file_methods_data(file):
    """
    Extract the source code tokens, identifier names and graph for methods in a source file
    represented by a graph. Identifier tokens are split into subtokens. Constructors are not
    included in the methods.
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

                # start_line_number = node.startLineNumber
                # end_line_number = node.endLineNumber
                # method_source = []
                # for other_node in g.node:
                #     if other_node.startLineNumber >= start_line_number and other_node.endLineNumber \
                #             <= end_line_number:
                #         # if other_node.type == FeatureNode.TOKEN:
                #         #     method_source.append(other_node.contents)
                #         # elif other_node.type == FeatureNode.IDENTIFIER_TOKEN:
                #         #     sub_identifiers = split_identifier_into_parts(other_node.contents)
                #         #     method_source += sub_identifiers
                #         if other_node.id == method_name_node.id:
                #             method_source.append('_')
                #         elif other_node.type == FeatureNode.TOKEN or other_node.type == \
                #                 FeatureNode.IDENTIFIER_TOKEN:
                #             method_source.append(other_node.contents)

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
    method_nodes_ids.append(node_id)

    for edge in file_adj_list[node_id]:
        if edge['edge_type'] != FeatureEdge.NEXT_TOKEN and edge['destination'] not in method_nodes_ids:
            get_method_nodes_rec(edge['destination'], method_nodes_ids, file_adj_list)


def remap_edges(edges, nodes):
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


def get_tokens(g):
    """
    Get the tokens for a file. Identifiers are split in subtokens.
    :param g: graph representing the file
    :return: list of tokens
    """
    token_nodes = list(filter(lambda n: n.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN),
                         g.node))
    tokens = []
    for token_node in token_nodes:
        if token_node.type == FeatureNode.IDENTIFIER_TOKEN:
            sub_identifiers = split_identifier_into_parts(token_node.contents)
            tokens += sub_identifiers
        else:
            tokens.append(token_node.contents)

    return tokens


def get_method_name_node(g, method_node):
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
    nx_graph = nx.DiGraph()
    with file.open('rb') as f:
        g = Graph()
        g.ParseFromString(f.read())

        for edge in g.edge:
            edge_type = [name for name, value in list(vars(FeatureEdge).items())[8:] if value ==
                         edge.type][0]
            nx_graph.add_edge(edge.sourceId, edge.destinationId, edge_type=edge_type)
    return nx_graph


def get_tokens_dataset_from_dir(dir="../corpus/r252-corpus-features/"):
    methods_source = []
    methods_names = []
    methods_graphs = []

    proto_files = list(Path(dir).rglob("*.proto"))
    print("A total of {} files have been found".format(len(proto_files)))

    # proto_files = [Path("../features-javac-master/Test.java.proto")]

    for i, file in enumerate(proto_files):
        # nx_graph = get_nx_graph(file)
        if i % 10 == 0:
            print("Extracting data from file {}".format(i+1))
        file_methods_source, file_methods_names, file_methods_graph = \
            get_file_methods_data(file)
        methods_source += file_methods_source
        methods_names += file_methods_names
        methods_graphs += file_methods_graph

    return methods_source, methods_names, methods_graphs


def get_method_nodes(method_node, file_graph):
    method_nodes = [method_node]
    get_method_nodes_rec(method_node, file_graph, method_nodes)

    return method_nodes


# def get_method_nodes_rec(node, file_graph, method_nodes):
#     print(len(method_nodes))
#     for e in file_graph.edge:
#         neighbour = e.destinationId
#         if neighbour not in method_nodes:
#             method_nodes.append(neighbour)
#             get_method_nodes(neighbour, nx_graph, method_nodes)


def get_augmented_graph(file):
    # TODO: Does each method in a file have a different graph?
    with file.open('rb') as f:
        g = Graph()
        g.ParseFromString(f.read())

        augmented_graph = nx.Graph()
        new_node_id = max([node.id for node in g.node]) + 1

        split_identifiers_node = [node for node in g.node if node.type == FeatureNode.IDENTIFIER_TOKEN
                                  and len(split_identifier_into_parts(node.contents)) > 1]

        # Add all edges
        for edge in g.edge:
            edge_type = [name for name, value in list(vars(FeatureEdge).items())[8:] if value ==
                         edge.type][0]
            augmented_graph.add_edge(edge.sourceId, edge.destinationId, edge_type=edge_type)

        # Add new edges for split identifiers and sub identifiers
        for node in split_identifiers_node:
            sub_identifiers = split_identifier_into_parts(node.contents)
            sub_identifiers_ids = list(range(new_node_id, new_node_id + len(sub_identifiers)))
            new_node_id += len(sub_identifiers)

            # ADD NEXT_TOKEN edge from node before identifier to first sub-identifier
            previous_token_node_id = find_previous_token_node_id(node, g)
            augmented_graph.add_edge(previous_token_node_id, sub_identifiers_ids[0],
                                     edge_type="NEXT_TOKEN")

            # ADD NEXT_TOKEN edge from last sub-identifier to node after identifier
            next_token_node_id = find_next_token_node_id(node, g)
            augmented_graph.add_edge(sub_identifiers_ids[-1], next_token_node_id,
                                     edge_type="NEXT_TOKEN")

            # ADD AST_CHILD edge from ast parent of node to first sub-identifier
            # ast_parent_node_id = find_ast_parent_node_id(node, g)
            # augmented_graph.add_edge(ast_parent_node_id, sub_identifiers_ids[0],
            #                          edge_type="ASSOCIATED_TOKEN")

            for i, sub_identifier_id in enumerate(sub_identifiers_ids):
                # Add IN_TOKEN edges from sub-identifiers to identifier
                augmented_graph.add_edge(sub_identifier_id, node.id, edge_type="IN_TOKEN")

                # ADD NEXT_TOKEN edges from sub-identifier to next sub-identifier
                if i < len(sub_identifiers_ids) - 1:
                    augmented_graph.add_edge(sub_identifiers_ids[i], sub_identifiers_ids[i + 1],
                                             edge_type="NEXT_TOKEN")
    return augmented_graph


def find_previous_token_node_id(node, g):
    for edge in g.edge:
        if edge.destinationId == node.id and edge.type == FeatureEdge.NEXT_TOKEN:
            return edge.sourceId

    return None


def find_next_token_node_id(node, g):
    for edge in g.edge:
        if edge.sourceId == node.id and edge.type == FeatureEdge.NEXT_TOKEN:
            return edge.destinationId

    return None


def find_ast_parent_node_id(node, g):
    for edge in g.edge:
        if edge.destinationId == node.id and edge.type == FeatureEdge.ASSOCIATED_TOKEN:
            return edge.sourceId

    return None
