def create_rgraph_data(data, features, max_num_nodes, max_num_edges):
    """
    Create a PyG data for RGCN from input data and features.

    Args:
        data (dict): Input data containing features.
        features (list): List of features to extract data from.
        max_num_nodes (int, optional): Maximum number of nodes in the dataset.
        max_num_edges (int, optional): Maximum number of edges in the dataset.

    Returns:
        torch_geometric.data.Data: PyG data for RGCN.
    """
    
    # Initialize node, edge, and edge weight lists
    nodes = [0] * max_num_nodes
    attention_mask = [0] * max_num_nodes
    edges = set()

    # Use defaultdict to easily create an empty list dictionary
    node_id_to_index = defaultdict(lambda: len(node_id_to_index))

    # Extract edges and edge weights from input data
    for feature in features:
        if 'edges' in feature:
            quadruples = data[feature]
            if 'parent' in feature:
                weights = data['parent_edges_weight']
            else:
                weights = [1] * len(quadruples)
            
            for quadruple, weight in zip(quadruples, weights):
                src, edge_type, dest, _ = quadruple
                
                if len(node_id_to_index) > max_num_nodes and (src not in node_id_to_index or dest not in node_id_to_index):
                    continue
                
                # Add node index to the node dictionary
                src_index = node_id_to_index[src]
                dest_index = node_id_to_index[dest]
                
                # Add edge to edges
                edges.add((src_index, dest_index, edge_type, weight))
    
    # Build node list and attention token mask
    for node, index in node_id_to_index.items():
        nodes[index] = node
        attention_mask[index] = 1
    
    # If number of edges is less than max_num_edges, pad with zeros
    edges = list(edges)
    edges += [(0, 0, 3, 0)] * (max_num_edges - len(edges))

    # Truncate edges and edge weight lists to match max_num_edges
    edges = edges[:max_num_edges]
    
    # Create PyG data object and return
    return Data(
        x=torch.tensor(nodes),
        edge_index=torch.tensor([edge[:2] for edge in edges], dtype=torch.long).t().contiguous(),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        edge_type=torch.tensor([edge[2] for edge in edges], dtype=torch.long),
        edge_weight=torch.tensor([edge[3] for edge in edges], dtype=torch.float),
        label=torch.tensor(data['labels'], dtype=torch.long)
    )