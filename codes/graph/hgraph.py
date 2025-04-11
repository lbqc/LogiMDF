import numpy as np
from collections import defaultdict


def build_hyperedge_index_from_node_set(hyperedge):
    """
    Construct the edge index of a hypergraph based on the given representation using node sets.

    Args:
    - hyperedge (list of sets): A list representing the hypergraph using node sets, where each set represents a hyperedge.

    Returns:
    - nodes (list): A list of unique nodes in the hypergraph.
    - hyperedge_index (numpy array): A numpy array containing the node indices and hyperedge indices of the hypergraph, 
        with shape (2, num_edges), 
        where the first row represents node indices and the second row represents corresponding hyperedge indices.
    """

    # Initialize lists to store hypergraph nodes and indices
    hyperedge_nodes = []
    hyperedge_indices = []
    nodes = []
    node2index = {}
    
    # Iterate through each hyperedge and its corresponding nodes
    for i, node_set in enumerate(hyperedge):
        # Extend the lists with hyperedge indices and nodes
        hyperedge_indices.extend([i] * len(node_set))
        node_set = sorted(list(node_set))
        for node in node_set:
            # If the node is not in node2index, add it and assign a unique index
            if node not in node2index:
                node2index[node] = len(node2index)
                nodes.append(node)
            hyperedge_nodes.append(node2index[node])
    
    return nodes, np.array([hyperedge_nodes, hyperedge_indices])


def extend_hyperedge_index_with_tokenization(tokenizer, nodes, edges):
    """
    Extend the hypergraph edge index with tokenization.

    Parameters:
    - tokenizer (Tokenizer): The tokenizer to use for encoding.
    - nodes (list): A list of unique nodes in the hypergraph.
    - edges (list): A list containing two lists: the first list contains node indices, and the second list contains hyperedge indices.

    Returns:
    - hyperedge_index (numpy array): An array where the first row represents the encoded nodes, 
        and the second row represents the corresponding hyperedge indices.
    """
    # Get unique nodes and encode them
    nodes_new = []
    node2encoded = [tokenizer.encode(node, add_special_tokens=False) for node in nodes]
    offset = [len(node) for node in node2encoded]
    num = 0
    offset_list = []
    for i in range(len(offset)):
        offset_list.append([num + ii for ii in range(len(node2encoded[i]))])
        num += offset[i]
        nodes_new += node2encoded[i]
    
    # Initialize lists to store encoded nodes and hyperedge indices
    hyperedge_index_node = []
    hyperedge_index_edge = []
    
    # Iterate through each node and its corresponding hyperedge index
    for node, edge_index in zip(edges[0], edges[1]):
        # Get the encoded representation of the node
        node_encoded = node2encoded[node]     
        # Extend the lists with encoded nodes and their corresponding hyperedge indices
        hyperedge_index_node.extend(offset_list[node])
        hyperedge_index_edge.extend([edge_index] * len(node_encoded))
        # nodes.extend(node_encoded)
    
    return np.asarray(nodes_new), np.asarray([hyperedge_index_node, hyperedge_index_edge])
    

def find_root_parent(node, parent):
    """
    Find the root parent of a node using the parent array.
    """
    key = node
    while parent[key] != -1:
        key = parent[key]
    if key != node:
        parent[node] = key
    return key


def build_hgraph_from_single_type_edges(nodes, edges):
    """
    Build a hypergraph from nodes and edges of a single type.
    """
    nodes = sorted(list(nodes))
    # Create a mapping from nodes to their indices
    
    node2index = {node: i for i, node in enumerate(nodes)}
    # print(node2index)
    
    # Initialize parent list where each node is initially its own parent
    child2parent = [-1] * len(nodes)
    
    # Union operation based on edges
    for src, dest in edges:
        # print(f'{src}, {dest}')
        if src not in node2index:
            node2index[src] = len(node2index)
            nodes.append(src)
            child2parent.append(-1)
        if dest not in node2index:
            node2index[dest] = len(node2index)
            nodes.append(dest)
            child2parent.append(-1)
        src_index = find_root_parent(node2index[src], child2parent)
        dest_index = find_root_parent(node2index[dest], child2parent)
        if src_index == dest_index:
            continue
        child2parent[dest_index] = src_index  # Union operation
    
    # Organize nodes into hyperedges
    parent_to_children = {}
    for i, parent_index in enumerate(child2parent):
        parent_index = find_root_parent(parent_index, child2parent)
        if parent_index not in parent_to_children:
            parent_to_children[parent_index] = {nodes[i]}
        else:
            parent_to_children[parent_index].add(nodes[i])
    
    return [children for children in parent_to_children.values() if len(children) > 1] 


def build_hypergraph_from_simple_graph(nodes, edges):
    """
    Build a hypergraph from nodes and edges of multiple types.
    There is room for further optimization.
    """
    type2edges = defaultdict(list)
    for src, dest, edge_type in edges:
        type2edges[edge_type].append((src, dest))
    
    hyper_edge_nodes = []
    hyper_edge_types = []
    for edge_type, edge_list in type2edges.items():
        sub_graphs = build_hgraph_from_single_type_edges(nodes, edge_list)
        hyper_edge_types.extend([edge_type] * len(sub_graphs))
        hyper_edge_nodes.extend(sub_graphs)
    
    return hyper_edge_nodes, hyper_edge_types
