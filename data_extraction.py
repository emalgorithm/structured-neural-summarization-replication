from graph_pb2 import FeatureNode
from text_util import split_identifier_into_parts


def get_tokens(g):
    # g.node is list of nodes
    token_nodes = list(filter(lambda n: n.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN),
                         g.node))
    # tokens_content = [token.contents for token in tokens]
    tokens = []
    for token_node in token_nodes:
        if token_node.type == FeatureNode.IDENTIFIER_TOKEN:
            sub_identifiers = split_identifier_into_parts(token_node.contents)
            tokens += sub_identifiers
        else:
            tokens.append(token_node.contents)

    return tokens
