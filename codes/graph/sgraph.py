import networkx as nx
import numpy as np


def get_line_graph_edges(data, attribute='type'):
    """
    Get the edges of the line graph from the given graph data.

    Args:
    - data: The input graph data. It can be either a NetworkX DiGraph or a list of edges.
    - attribute: The attribute to consider for edge mapping.

    Returns:
    - node_list: List of unique nodes in the line graph.
    - edge_list: List of edges in the line graph.
    """
    if isinstance(data, nx.DiGraph):
        graph = data
    else:
        graph = nx.DiGraph()
        graph.add_edges_from(data)

    mapping = {(src, dst): attr[attribute] for src, dst, attr in graph.edges(data=True)}

    result = set()
    line_graph = nx.line_graph(graph)
    for edge in line_graph.edges:
        src, dst = edge
        attr = list(set(src) & set(dst))[0]  # TODO: Simplified
        result.add((mapping[src], mapping[dst], attr))
    
    return list(set(mapping.values())), list(result)
