from graph_pb2 import FeatureNode, FeatureEdge
from text_util import split_identifier_into_parts
from pathlib import Path
from graph_pb2 import Graph
import networkx as nx


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


def get_methods_source_and_name(file):
    """
    Extract the source code token and identifier names for methods in a source file represented
    by a graph. Identifier tokens are split into subtokens. Constructors are not included in the
    methods.
    :param file: file
    :return: (methods_source, methods_names) where methods_source[i] is a list of the tokens for
    the source of ith method in the file, and methods_names[i] is a list of tokens for name of the
    ith
    """
    with file.open('rb') as f:
        class_name = file.name.split('.')[0]

        g = Graph()
        g.ParseFromString(f.read())
        methods_source = []
        methods_names = []
        # class_name_node = get_class_name_node(g)

        for node in g.node:
            if node.contents == "METHOD":
                method_name_node = get_method_name_node(g, node)

                # If method name is the same as class name, then method name is constructor,
                # so discard it
                if method_name_node.contents == class_name:
                    continue

                methods_names.append(split_identifier_into_parts(method_name_node.contents))

                start_line_number = node.startLineNumber
                end_line_number = node.endLineNumber
                method_source = []
                for other_node in g.node:
                    if other_node.startLineNumber >= start_line_number and other_node.endLineNumber \
                            <= end_line_number and other_node.id != method_name_node.id:
                        if other_node.type == FeatureNode.TOKEN:
                            method_source.append(other_node.contents)
                        elif other_node.type == FeatureNode.IDENTIFIER_TOKEN:
                            sub_identifiers = split_identifier_into_parts(other_node.contents)
                            method_source += sub_identifiers

                methods_source.append(method_source)

        return methods_source, methods_names


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


def get_dataset_from_dir(dir="../corpus/r252-corpus-features/"):
    methods_source = []
    methods_names = []
    method_graphs = []

    proto_files = list(Path(dir).rglob("*.proto"))
    print("A total of {} files have been found".format(len(proto_files)))

    for i, file in enumerate(proto_files):
        if i % 100 == 0:
            print("Extracting data from file {}".format(i+1))
        file_methods_source, file_methods_names = get_methods_source_and_name(file)
        methods_source += file_methods_source
        methods_names += file_methods_names
        method_graph = get_augmented_graph(file)
        method_graphs += [method_graph for _ in range(len(methods_source))]

    return methods_source, methods_names, method_graphs


def get_augmented_graph(file):
    # TODO: Does each method in a file have a different graph?
    with file.open('rb') as f:
        g = Graph()
        g.ParseFromString(f.read())

        augmented_graph = nx.Graph()
        new_node_id = max([node.id for node in g.node]) + 1

        split_identifiers_node = [node for node in g.node if node.type == FeatureNode.IDENTIFIER_TOKEN
                                  and len(split_identifier_into_parts(node.contents)) > 1]
        split_identifiers_node_ids = [node.id for node in split_identifiers_node]

        # Add all edges apart from the ones to and from split identifiers
        for edge in g.edge:
            if edge.sourceId not in split_identifiers_node_ids and edge.destinationId not in \
                    split_identifiers_node_ids:
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
            ast_parent_node_id = find_ast_parent_node_id(node, g)
            augmented_graph.add_edge(ast_parent_node_id, sub_identifiers_ids[0],
                                     edge_type="ASSOCIATED_TOKEN")

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
