import networkx as nx
import numpy as np


def extend_adjacency_matrix_with_tokenization(tokenizer, edges, weight='type', self_loop=4):
    """
    Extend an adjacency matrix by tokenizing node descriptions and inserting them as new rows and columns.
    
    Args:
    - tokenizer (Tokenizer): Tokenizer object used to encode node descriptions.
    - edges (((src, dest, {edge_attr}), ...) or nx.DiGraph): NetworkX graph containing the edges.
    - weight (str): Attribute of the edges to use as weights for the adjacency matrix (default is 'type').
    - self_loop (int): Default value to set diagonal elements of the adjacency matrix (default is 4).
    
    Returns:
    - graph_node_encoded (numpy.ndarray): Array containing the token indices for the extended graph.
    - adjacency_matrix (numpy.ndarray): Extended adjacency matrix representing the graph.
    """
    
    # Initialize directed graph and add edges
    if isinstance(edges, nx.DiGraph):
        graph = edges
    else:
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
    
    # Get nodes and convert graph to adjacency matrix
    nodes = graph.nodes
    adjacency_matrix = nx.to_numpy_array(graph, weight=weight)
    
    # Set diagonal elements to a default value
    np.fill_diagonal(adjacency_matrix, self_loop)
    
    # Initialize index for following graph and list to store token indices
    row_index = 0
    graph_node_encoded = []
    
    # Iterate over nodes to encode and extend the adjacency matrix
    for _, desc in enumerate(nodes):
        encoded_desc = tokenizer.encode(desc, add_special_tokens=False)
        graph_node_encoded.extend(encoded_desc)
        
        length = len(encoded_desc)
        
        # Extend adjacency matrix with new rows and columns
        while length > 1:
            new_row = adjacency_matrix[row_index, :]
            adjacency_matrix = np.insert(adjacency_matrix, row_index, new_row, axis=0)

            new_col = adjacency_matrix[:, row_index]
            adjacency_matrix = np.insert(adjacency_matrix, row_index, new_col, axis=1)
            
            length -= 1
            row_index += 1
        
        row_index += 1
    
    return np.asarray(graph_node_encoded), np.asarray(adjacency_matrix)


def adjacency_matrix_to_quadruples(nodes, adjacency_matrix, timestamp):
    """
    Convert an adjacency matrix to a list of quadruples representing the edges.
    
    Parameters:
    - nodes (list): List of node labels.
    - adjacency_matrix (np.ndarray): Adjacency matrix of the graph.
    - timestamp (int): Order of the graph.
    
    Returns:
    - quadruples (numpy.ndarray): quadruples representing the edges, in the format (source, target, weight, timestamp).
    """
    
    # Get indices of non-zero entries in the adjacency matrix
    edge_indices = np.transpose(np.nonzero(adjacency_matrix))
    
    # Extract edge values corresponding to the edge indices
    edge_values = adjacency_matrix[edge_indices[:, 0], edge_indices[:, 1]]
    
    # Create quadruples from edge indices and values
    quadruples = [(nodes[index[0]], nodes[index[1]], value - 1, timestamp) for index, value in zip(edge_indices, edge_values)]
    
    return np.asarray(quadruples)
