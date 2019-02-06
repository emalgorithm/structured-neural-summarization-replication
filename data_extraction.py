from graph_pb2 import FeatureNode, FeatureEdge
from text_util import split_identifier_into_parts


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


def get_methods_source_and_name(g):
    # TODO: Discard constructors methods
    """
    Extract the source code token and identifier names for methods in a source file represented
    by a graph. Identifier tokens are split into subtokens. Constructors are not included in the
    methods.
    :param g: graph representing the file
    :return: (methods_source, methods_names) where methods_source[i] is a list of the tokens for
    the source of ith method in the file, and methods_names[i] is a list of tokens for name of the
    ith
    """
    methods_source = []
    methods_names = []

    for node in g.node:
        if node.contents == "METHOD":
            method_name_node = get_method_name_node(g, node)
            methods_names.append(split_identifier_into_parts(method_name_node.contents))

            # If method name is the same as class name, then method name is constructor,
            # so discard it
            class_name_node = get_class_name_node(g)
            if method_name_node.contents != class_name_node.contents:
                continue

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
    return 0
