from graph_pb2 import FeatureNode, FeatureEdge
from text_util import split_identifier_into_parts
from pathlib import Path
from graph_pb2 import Graph


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

    proto_files = list(Path(dir).rglob("*.proto"))
    print("A total of {} files have been found".format(len(proto_files)))

    for i, file in enumerate(proto_files):
        print("Extracting data from file {}".format(i+1))
        file_methods_source, file_methods_names = get_methods_source_and_name(file)
        methods_source += file_methods_source
        methods_names += file_methods_names

    return methods_source, methods_names
