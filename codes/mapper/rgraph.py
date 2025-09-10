import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

from data.file_io import load_data, load_model
from graph.rgraph import extend_adjacency_matrix_with_tokenization, adjacency_matrix_to_quadruples
from data.fol import bulid_graph_from_fol
from cluster.predict import predict_cls_name, predict_category_name
import os
import pickle


class FOLRGraphMapper(object):
    def __init__(self, tokenizer, stopwords, parent_graph_config, label_mapper=None, max_num_nodes=25, max_num_edges=300, parent_need=False, child2paren_need=False):
        """
        Initialize FOLMapper with necessary configurations and data.
        
        Parameters:
        - tokenizer: Tokenizer for text encoding.
        - stopwords: List of stopwords to filter out from text.
        - label_mapper: Dictionary for mapping labels to numerical values.
        - parent_graph_config: Configuration dictionary for parent graph.
        """
        # Load data from configuration
        if isinstance(parent_graph_config, str):
            parent_graph_config = load_data(parent_graph_config)
        
        self.relation2index = parent_graph_config.get('relation2index', {
            '∧': 4, 
            '∨': 3, 
            '→': 2,
            'self-loop': 1,
            'child2parent': 5,
        })
        self.relation_counter = [0] * len(self.relation2index)
        
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.label_mapper = label_mapper
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.parent_need = parent_need
        self.child2parent_need = child2paren_need
        
        if isinstance(self.stopwords, str):
            self.stopwords = load_data(self.stopwords)
        
        parent_graph_nodes = load_data(parent_graph_config['node'])['name'].values.squeeze()
        parent_graph_edges = load_data(parent_graph_config['edge'])
        group_index = (parent_graph_edges.index // 3)
        parent_graph_edges.loc[:, 'weight'] = parent_graph_edges.groupby(group_index)['weight'].transform(lambda x: x / x.sum())
        # print(parent_graph_edges.loc[100:150, 'weight'])
        parent_graph_edges = parent_graph_edges[parent_graph_edges['weight'] > 0].values.squeeze()
        self.cls2name = load_data(parent_graph_config['cls2name'])['name'].values.squeeze()
        
        # Load parent graph classifier, tokenizer, and model
        self.parent_graph_classifier = load_model(parent_graph_config['classifier'])
        self.parent_graph_tokenizer = AutoTokenizer.from_pretrained(parent_graph_config['pretrained_model_name'])
        self.parent_graph_model = AutoModel.from_pretrained(parent_graph_config['pretrained_model_name'])
        self.parent_graph_model = self.parent_graph_model.cuda()
        print(torch.cuda.is_available())
        
        # Set other configurations
        self.max_node_description_length = parent_graph_config.get('max_node_description_length', 8)
        
        self.predict_cache_filename = './cls_cache.pkl'
        self.fol_cache_filename = f'./rgraph_fol_cache_{self.parent_need}_{self.child2parent_need}_{self.max_num_nodes}_{self.max_num_edges}.pkl'
        self.n_hop_cache_filename = './3_hop.pkl'
        print(f'fol: {self.fol_cache_filename}')
        self.predict_cache = {}
        if os.path.exists(self.predict_cache_filename):
            with open(self.predict_cache_filename, "rb") as f:
                self.predict_cache = pickle.load(f)
        self.fol_cache = {}
        if os.path.exists(self.fol_cache_filename):
            with open(self.fol_cache_filename, "rb") as f:
                self.fol_cache = pickle.load(f)
        self.n_hop_cache = {}
        if os.path.exists(self.n_hop_cache_filename):
            with open(self.n_hop_cache_filename, "rb") as f:
                self.fol_cache = pickle.load(f)
        self.changed = False
        # Create and populate parent graph
        self.parent_graph = nx.DiGraph()
        self.parent_graph.add_edges_from([(src, dest, {f'type{self._relation2index(edge_type)}': 1 + weight}) for src, dest, edge_type, weight in parent_graph_edges])
        self.parent_weight_type = [f'type{i}' for i in range(1, 4)]
        
        # Create label graph
        opposed = parent_graph_config.get('label_opposed_fol', '( ( saddening ∨ opposed ∨ dislike ∨ annoying ∨ disgusting ∨ sad ) ∨ ( not & support & neutral ) ) → opposed')
        support = parent_graph_config.get('label_support_fol', '( ( support ∨ acculturation ∨ recommend ∨ acclaim ∨ stimulate_mind ∨ endorse ) ∨ ( not & opposed & neutral ) ) → support')
        neutral = parent_graph_config.get('label_neutral_fol', '( ( inactive ∨ inert ∨ stable ∨ neutral ∨ indifferent ∨ unreactive ) ∨ ( not & support & opposed ) ) → neutral')
        
        label_graph_edges_list = [bulid_graph_from_fol(fol, self.stopwords)[1] for fol in [opposed, support, neutral]]
        label_graph_edges_list = [[(src, dest, {'type': self._relation2index(edge_type)}) for src, dest, edge_type in label_edges] for label_edges in label_graph_edges_list]
        label_nodes_and_adj_matrix = [extend_adjacency_matrix_with_tokenization(self.tokenizer, edges_with_types) for edges_with_types in label_graph_edges_list]
        self.label_node_encoded = [item[0] for item in label_nodes_and_adj_matrix]
        self.label_adj_matrix = [item[1] for item in label_nodes_and_adj_matrix]
        self.timestep = 1
    
    def save_cache(self):
        print(self.changed)
        if not os.path.exists(self.predict_cache_filename) or self.changed:
            with open(self.predict_cache_filename, 'wb') as f :
                pickle.dump(self.predict_cache, f)
        with open(self.n_hop_cache_filename, "wb") as f:
                pickle.dump(self.n_hop_cache, f)
        # if not os.path.exists(self.fol_cache_filename) or self.changed:
        #     with open(self.fol_cache_filename, 'wb') as f:
        #         pickle.dump(self.fol_cache, f) 
    
    def __call__(self, batched_input):
        """
        Process batched input to generate various graph-related data.
        
        Parameters:
        - batched_input (dict): Dictionary containing batched input data.
        
        Returns:
        - dict: Dictionary containing processed graph-related data.
        """
        
        # Initialize result dictionary
        processed_data = {
            'parent_nodes': [],
            'parent_edges': [],
            'parent_edges_weight': [],
            'child_nodes': [],
            'child_edges': [],
            'child2parent_nodes': [],
            'child2parent_edges': [],
            'label_nodes': [],
            'label_edges': [],
            'labels': [],
        }
        
        for fol, label_str in zip(batched_input['FOL'], batched_input['Stance']):
            # Extract nodes and edges for the current FOL
            child_nodes, child_edges = bulid_graph_from_fol(fol, self.stopwords)
            
            # Convert child edges to include edge types
            child_edges_with_types = [(src, dest, {'type': self._relation2index(edge_type)}) for src, dest, edge_type in child_edges]
            
            # Extend adjacency matrix and generate quadruples for child graph
            child_node_encoded, child_adj_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, child_edges_with_types, self_loop=self._relation2index('self-loop'))
            child_quadruples = adjacency_matrix_to_quadruples(child_node_encoded, child_adj_matrix, self.timestep + 2)
            
            # Generate child-to-parent mapping
            child2parent_mapping = {}
            for node in child_nodes:
                parent = predict_cls_name(node, self.parent_graph_tokenizer, self.parent_graph_model, self.parent_graph_classifier, self.cls2name, self.predict_cache, self.max_node_description_length)
                child2parent_mapping[node] = parent
            
            # Generate child2parent graph
            child2parent_edges = [(src, dest, {"type": self._relation2index('child2parent')}) for src, dest in child2parent_mapping.items()]
            encoded_child2parent, child2parent_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, child2parent_edges, self_loop=self._relation2index('self-loop'))
            child2parent_quadruples = adjacency_matrix_to_quadruples(encoded_child2parent, child2parent_matrix, self.timestep + 1)
            
            # Create parent subgraph and generate adjacency matrix and quadruples
            parent_subgraph_nodes = set(child2parent_mapping.values())
            parent_subgraph_nodes = set(nx.single_source_shortest_path_length(self.parent_graph.subgraph, parent_subgraph_nodes, cutoff=2).keys())
            parent_quadruples = []
            parent_edge_weight = []
            for i, weight_type in enumerate(self.parent_weight_type):
                encoded_parent_nodes, parent_adj_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, self.parent_graph.subgraph(parent_subgraph_nodes), weight=weight_type, self_loop=2)
                quadruples = adjacency_matrix_to_quadruples(encoded_parent_nodes, parent_adj_matrix, self.timestep)
                parent_edge_weight += [quadruple[1] for quadruple in quadruples]
                parent_quadruples += [(quadruple[0], i - 1 if quadruple[1] < 1 else self._relation2index('self-loop') - 1, quadruple[2], quadruple[3]) for quadruple in quadruples]
            parent_edge_weight = np.asarray(parent_edge_weight)
            parent_quadruples = np.asarray(parent_quadruples)
            
            label = self.label_mapper[label_str]
            encoded_label_nodes = self.label_node_encoded[label]
            label_quadruples = adjacency_matrix_to_quadruples(encoded_label_nodes, self.label_adj_matrix[label], self.timestep + 3)
            
            # Append results to the result dictionary
            processed_data['parent_nodes'].append(encoded_parent_nodes)
            processed_data['parent_edges'].append(parent_quadruples)
            processed_data['parent_edges_weight'].append(parent_edge_weight)
            processed_data['child2parent_nodes'].append(encoded_child2parent)
            processed_data['child2parent_edges'].append(child2parent_quadruples)
            processed_data['child_nodes'].append(child_node_encoded)
            processed_data['child_edges'].append(child_quadruples)
            processed_data['label_nodes'].append(encoded_label_nodes)
            processed_data['label_edges'].append(label_quadruples)
            processed_data['labels'].append(label)

            self.timestep += 3

        return processed_data
    
    def _relation2index(self, relation):
        index = self.relation2index.get(relation, 4)
        self.relation_counter[index - 1] += 1
        return index
    
    
    def generate_graph_data(self, fol, parent_need=True, child2parent_need=False):
        if fol in self.fol_cache:
            return self.fol_cache[fol]
        parent_need = self.parent_need
        child2parent_need = self.child2parent_need
        
        # Initialize result dictionary
        processed_data = {
            'parent_nodes': [],
            'parent_edges': [],
            'parent_edges_weight': [],
            'child_nodes': [],
            'child_edges': [],
            'child2parent_nodes': [],
            'child2parent_edges': [],
        }
        child_nodes, child_edges = bulid_graph_from_fol(fol, self.stopwords)
        type2child_graph = {}
        for src, dest, edge_type in child_edges:
            edge_type = self._relation2index(edge_type)
            if edge_type not in type2child_graph:
                type2child_graph[edge_type] = [(src, dest, {"type": edge_type})]
            else:
                type2child_graph[edge_type].append((src, dest, {"type": edge_type}))
        
        # Convert child edges to include edge types
        child_node_encoded = []
        child_quadruples = []
        for edges in type2child_graph.values():
            # Extend adjacency matrix and generate quadruples for child graph
            nodes, adj_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, edges, self_loop=self._relation2index('self-loop'))
            quadruples = adjacency_matrix_to_quadruples(nodes, adj_matrix, self.timestep + 2)
            child_node_encoded.append(nodes)
            child_quadruples.append(quadruples)
        child_node_encoded = np.concatenate(child_node_encoded, axis=0)
        child_quadruples = np.concatenate(child_quadruples, axis=0)
        # if len(child_node_encoded) > 0:
        #     child_node_encoded = np.concatenate(child_node_encoded, axis=0)
        #     child_quadruples = np.concatenate(child_quadruples, axis=0)
        # else:
        #     child_node_encoded = np.zeros(1)
        #     child_quadruples = np.zeros((1,4))
        
        # Generate child-to-parent mapping
        child2parent_mapping = {}
        for node in child_nodes:
            parent = predict_cls_name(node, self.parent_graph_tokenizer, self.parent_graph_model, self.parent_graph_classifier, self.cls2name, self.predict_cache, self.max_node_description_length)
            child2parent_mapping[node] = parent
        
        # Generate child2parent graph
        child2parent_edges = [(src, dest, {"type": self._relation2index('child2parent')}) for src, dest in child2parent_mapping.items()]
        encoded_child2parent, child2parent_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, child2parent_edges, self_loop=self._relation2index('self-loop'))
        child2parent_quadruples = adjacency_matrix_to_quadruples(encoded_child2parent, child2parent_matrix, self.timestep + 1)
        
        # Create parent subgraph and generate adjacency matrix and quadruples
        # parent_subgraph_nodes = set(child2parent_mapping.values())
        parent_subgraph_nodes = set(child2parent_mapping.values())
        # for node in child2parent_mapping.values():
        #     if node not in self.n_hop_cache:
        #         if node not in self.parent_graph:
        #             continue
        #         self.n_hop_cache[node] = nx.single_source_dijkstra(self.parent_graph, node, cutoff=3)[0]
        #     for tgt, distance in self.n_hop_cache[node].items():
        #         if len(parent_subgraph_nodes) > 24:
        #             break
        #         if distance <= 3:
        #             parent_subgraph_nodes.add(tgt)
        #     if len(parent_subgraph_nodes) > 24:
        #         break
        for node in child2parent_mapping.values():
            if node not in self.n_hop_cache:
                if node not in self.parent_graph:
                    continue
                self.n_hop_cache[node] = nx.single_source_dijkstra(self.parent_graph, node, cutoff=3)[0]
            for tgt, distance in self.n_hop_cache[node].items():
                if len(parent_subgraph_nodes) > 64:
                    break
                if distance <= 3:
                    if self.parent_graph.has_edge(node, tgt):
                        edge_xxx = self.parent_graph[node][tgt]
                        if 'type2' in edge_xxx and edge_xxx['type2'] > 0:
                            parent_subgraph_nodes.add(tgt)
            if len(parent_subgraph_nodes) > 64:
                break
        parent_quadruples = []
        parent_edge_weight = []
        for i, weight_type in enumerate(self.parent_weight_type):
            encoded_parent_nodes, parent_adj_matrix = extend_adjacency_matrix_with_tokenization(self.tokenizer, self.parent_graph.subgraph(parent_subgraph_nodes), weight=weight_type, self_loop=2)
            quadruples = adjacency_matrix_to_quadruples(encoded_parent_nodes, parent_adj_matrix, self.timestep)
            parent_edge_weight += [quadruple[2] for quadruple in quadruples]
            parent_quadruples += [(quadruple[0], quadruple[1], i if quadruple[2] < 1 else self._relation2index('self-loop') - 1, quadruple[3]) for quadruple in quadruples]
        parent_edge_weight = np.asarray(parent_edge_weight)
        parent_quadruples = np.asarray(parent_quadruples)
        
        self.timestep += 3
        
        # Append results to the result dictionary
        processed_data['parent_nodes'] = np.asarray(encoded_parent_nodes, dtype=np.int64)
        processed_data['parent_edges'] = np.asarray(parent_quadruples, dtype=np.int64)
        processed_data['parent_edges_weight'] = parent_edge_weight
        processed_data['child2parent_nodes'] = np.asarray(encoded_child2parent, dtype=np.int64)
        processed_data['child2parent_edges'] = np.asarray(child2parent_quadruples, dtype=np.int64)
        processed_data['child_nodes'] = np.asarray(child_node_encoded, dtype=np.int64)
        processed_data['child_edges'] = np.asarray(child_quadruples, dtype=np.int64)
        
        features = ['child_nodes', 'child_edges']
        # features = ['parent_nodes', 'parent_edges', 'parent_edges_weight']
        if parent_need:
            features.extend(['parent_nodes', 'parent_edges', 'parent_edges_weight'])
        if child2parent_need:
            features.extend(['child2parent_nodes', 'child2parent_edges'])
        data = {feature: processed_data[feature] for feature in features}
        result = self._format_rgraph_data(data, features)
        self.fol_cache[fol] = result
        self.changed = True
        return result
        
    def _format_rgraph_data(self, data, features, max_num_nodes=None, max_num_edges=None):
        """
        Create a PyG data for RGCN from input data and features.

        Args:
            data (dict): Input data containing features.
            features (list): List of features to extract data from.
            max_num_nodes (int, optional): Maximum number of nodes in the dataset.
            max_num_edges (int, optional): Maximum number of edges in the dataset.

        Returns:
            edge_index, edge_type, x
        """
        if max_num_nodes is None:
            max_num_nodes = self.max_num_nodes
        if max_num_edges is None:
            max_num_edges = self.max_num_edges
        
        # Initialize node, edge, and edge weight lists
        nodes = [0] * max_num_nodes
        attention_mask = [0] * max_num_nodes
        edges = set()

        # Use defaultdict to easily create an empty list dictionary
        node_id_to_index = defaultdict(lambda: len(node_id_to_index))

        # Extract edges and edge weights from input data
        # print(features)
        for feature in features:
            if 'edges' in feature and 'edges_' not in feature:
                quadruples = data[feature]
                # print('quadruples')
                # print(quadruples)
                if 'parent_edges_weight' in feature:
                    weights = data['parent_edges_weight']
                else:
                    weights = [1] * len(quadruples)
                
                for quadruple, weight in zip(quadruples, weights):
                    src, dest, edge_type, _ = quadruple
                    
                    if len(node_id_to_index) >= max_num_nodes and (src not in node_id_to_index or dest not in node_id_to_index):
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
        edges += [(0, 0, 0, 0)] * (max_num_edges - len(edges))

        # Truncate edges and edge weight lists to match max_num_edges
        edges = edges[:max_num_edges]
        
        return torch.tensor([edge[:2] for edge in edges], dtype=torch.long).t().contiguous(), torch.tensor([edge[-1:-3:-1] for edge in edges], dtype=torch.float).t().contiguous(), torch.tensor(nodes)
        