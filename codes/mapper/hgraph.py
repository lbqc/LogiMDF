import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data.file_io import load_data, load_model
from graph.hgraph import build_hypergraph_from_simple_graph, extend_hyperedge_index_with_tokenization, build_hyperedge_index_from_node_set
from graph.sgraph import get_line_graph_edges
from data.fol import bulid_hgraph_from_fol, bulid_graph_from_fol
from cluster.predict import predict_cls_name, predict_category_name
import os
import pickle

def build_cosine_similarity_hyperedges(nodes, tokenizer, encoder, threshold=0.5, max_length=32, encoder_output='pooler_output'):
    assert isinstance(nodes, list) and isinstance(nodes[0], str), f'{nodes}'
    node_encoded = tokenizer.batch_encode_plus(
        nodes, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        node_encoded = node_encoded.to(encoder.device)
        vectors = encoder(**node_encoded)[encoder_output]
    norm_vectors = vectors / vectors.norm(dim=1, keepdim=True)
    cosine_similarity_matrix = torch.mm(norm_vectors, norm_vectors.T)
    
    nodes_np = np.array(nodes)
    similar_phrases = []
    for i in range(cosine_similarity_matrix.size(0)):
        similar_indices = (cosine_similarity_matrix[i] > threshold).nonzero(as_tuple=True)[0]
        if len(similar_indices) > 1:
            similar_indices_np = similar_indices.cpu().numpy()
            similar_phrases.append(set(nodes_np[similar_indices_np].tolist()))
    
    return similar_phrases


class FOLHGraphMapper(object):
    def __init__(self, tokenizer, stopwords, parent_graph_config, label_mapper=None, max_num_nodes=25, max_num_edges=300, parent_need=False, child2paren_need=False):
        
        # Load data from configuration
        if isinstance(parent_graph_config, str):
            parent_graph_config = load_data(parent_graph_config)
        
        self.relation2description = parent_graph_config.get('relation2description', {
            '∧': 'and', 
            '∨': 'or', 
            '→': 'imply',
            'self-loop': 'self-loop',
            'child2parent': 'child to parent',
        })
        self.relation2index = parent_graph_config.get('relation2index', {
            '∧': 3, 
            '∨': 2, 
            '→': 1,
            'self-loop': 4,
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
        
        # Load data from configuration
        parent_graph_nodes = load_data(parent_graph_config['node'])['name'].values.squeeze()
        parent_graph_edges = load_data(parent_graph_config['edge'])
        parent_graph_edges = parent_graph_edges[parent_graph_edges['weight'] > 0].values.squeeze()
        self.cls2name = load_data(parent_graph_config['cls2name'])['name'].values.squeeze()
        
        # Load parent graph classifier, tokenizer, and model
        self.parent_graph_classifier = load_model(parent_graph_config['classifier'])
        self.parent_graph_tokenizer = AutoTokenizer.from_pretrained(parent_graph_config['pretrained_model_name'])
        self.parent_graph_model = AutoModel.from_pretrained(parent_graph_config['pretrained_model_name'])
        self.parent_graph_model = self.parent_graph_model.to('cuda')
        
        # Set other configurations
        self.max_node_description_length = parent_graph_config.get('max_node_description_length', 8)
        
        self.predict_cache_filename = './cls_cache.pkl'
        self.fol_cache_filename = f'./hgraph_fol_cache_{self.parent_need}_{self.child2parent_need}_{self.max_num_nodes}_{self.max_num_edges}.pkl'
        print(f'fol: {self.fol_cache_filename}')
        self.predict_cache = {}
        if os.path.exists(self.predict_cache_filename):
            with open(self.predict_cache_filename, "rb") as f:
                self.predict_cache = pickle.load(f)
        self.fol_cache = {}
        if os.path.exists(self.fol_cache_filename):
            with open(self.fol_cache_filename, "rb") as f:
                self.fol_cache = pickle.load(f)
        self.changed = False
        
        # Create parent hyper graph
        self.parent_hyperedge_nodes, self.parent_hyperedge_types = build_hypergraph_from_simple_graph(parent_graph_nodes, [(edge[0], edge[1], f'{edge[2]} {edge[3]}') for edge in parent_graph_edges])
        self.parent_node2hyperedges = {}
        for i, nodes in enumerate(self.parent_hyperedge_nodes):
            for node in nodes:
                if node not in self.parent_node2hyperedges:
                    self.parent_node2hyperedges[node] = {i}
                else:
                    self.parent_node2hyperedges[node].add(i)
                    
        # Create parent relation hyper graph
        type2parent_graph_edges_nx = {}
        for src, dst, edge_type, weight in parent_graph_edges:
            edge_type = self.relation2description[edge_type]
            if edge_type not in type2parent_graph_edges_nx:
                type2parent_graph_edges_nx[edge_type] = [(src, dst, {'weight': f'{edge_type} {weight}'})]
            else:
                type2parent_graph_edges_nx[edge_type].append((src, dst, {'weight': f'{edge_type} {weight}'}))

        parent_relation_graph_nodes, parent_relation_graph_edges = [], []
        for parent_graph_edges_nx in type2parent_graph_edges_nx.values():
            relation_graph = get_line_graph_edges(parent_graph_edges_nx, 'weight')
            parent_relation_graph_edges += relation_graph[1]
            parent_relation_graph_nodes += relation_graph[0]
        self.parent_relation_hyperedge_nodes, self.parent_relation_hyperedge_types = build_hypergraph_from_simple_graph(list(set(parent_relation_graph_nodes)), parent_relation_graph_edges)
        # print(self.parent_relation_hyperedge_nodes)
        # self.parent_relation_hyperedge_nodes, self.parent_relation_hyperedge_index = build_hyperedge_index_from_node_set(self.parent_relation_hyperedge_nodes)
        # self.parent_relation_hyperedge_nodes_encoded, self.parent_relation_hyperedge_index = extend_hyperedge_index_with_tokenization(self.tokenizer, self.parent_relation_hyperedge_nodes, self.parent_relation_hyperedge_index)
    
    def save_cache(self):
        print(self.changed)
        if not os.path.exists(self.predict_cache_filename) or self.changed:
            with open(self.predict_cache_filename, 'wb') as f:
                pickle.dump(self.predict_cache, f)
        # if not os.path.exists(self.fol_cache_filename) or self.changed:
        #     with open(self.fol_cache_filename, 'wb') as f:
        #         pickle.dump(self.fol_cache, f) 
        
    def __call__(self, batched_input):
        
        # Initialize result dictionary
        processed_data = {
            'parent_hypergraph_x': [], 
            'parent_hyperedge_index': [],
            'parent_relation_hypergraph_x': [],
            'parent_relation_hyperedge_index': [],
            'child_hypergraph_x': [],
            'child_hyperedge_index': [],
            # 'child_relation_hypergraph_x': [],
            # 'child_realtion_hyperedge_index': [],
            'labels': [],
        }
        
        for fol, label_str in zip(batched_input['FOL'], batched_input['Stance']):
            child_hyperedge_nodes, child_hyperedge_types = bulid_hgraph_from_fol(fol, self.stopwords)
            child_hyperedge_nodes, child_hyperedge_index = build_hyperedge_index_from_node_set(child_hyperedge_nodes)
            child_hyperedge_nodes_encoded, child_hyperedge_index = extend_hyperedge_index_with_tokenization(self.tokenizer, child_hyperedge_nodes, child_hyperedge_index)
            
            # child_relation_graph_nodes, child_relation_graph_edges = get_line_graph_edges(child_hyperedge_index)
            
            # Generate child-to-parent mapping
            child2parent_mapping = {}
            for node in child_hyperedge_nodes:
                parent = predict_cls_name(node, self.parent_graph_tokenizer, self.parent_graph_model, self.parent_graph_classifier, self.cls2name, self.predict_cache, self.max_node_description_length)
                child2parent_mapping[node] = parent
            
            parent_hyperedge_nodes = []
            parent_nodes = child2parent_mapping.values()
            parent_hyperedge_index_inculded = set()
            for node in parent_nodes:
                parent_hyperedge_index_inculded |= self.parent_node2hyperedges[node]
            parent_nodes = set(parent_nodes)
            for index in parent_hyperedge_index_inculded:
                nodes = self.parent_hyperedge_nodes[index] & parent_nodes
                if len(nodes) > 1:
                    parent_hyperedge_nodes.append(nodes)
            label = self.label_mapper[label_str]
            parent_hyperedge_nodes, parent_hyperedge_index = build_hyperedge_index_from_node_set(parent_hyperedge_nodes)
            parent_hyperedge_nodes_encoded, parent_hyperedge_index = extend_hyperedge_index_with_tokenization(self.tokenizer, child_hyperedge_nodes, child_hyperedge_index)
            
            processed_data['child_hypergraph_x'].append(child_hyperedge_nodes_encoded)
            processed_data['child_hyperedge_index'].append(child_hyperedge_index)
            processed_data['parent_hypergraph_x'].append(parent_hyperedge_nodes_encoded)
            processed_data['parent_hyperedge_index'].append(parent_hyperedge_index)
            processed_data['parent_relation_hypergraph_x'].append(self.parent_relation_hyperedge_nodes_encoded)
            processed_data['parent_relation_hyperedge_index'].append(self.parent_relation_hyperedge_index)
            processed_data['labels'].append(label)
            
        return processed_data
    
    def generate_hgraph_data(self, fol, parent_need=False, relation_needed=False, semantic_similarity_threshold=0):
        if fol in self.fol_cache:
            return self.fol_cache[fol]
        parent_need = self.parent_need
        relation_needed = self.child2parent_need
        child_hyperedge_nodes, child_hyperedge_types = bulid_hgraph_from_fol(fol, self.stopwords)
        child_rgraph_nodes, child_rgraph_edges = bulid_graph_from_fol(fol, self.stopwords)
        # child_rgraph_edges = [(src, dst, {'type': self.relation2description[edge_type]}) for src, dst, edge_type in child_rgraph_edges]
        # child_relation_nodes, child_relation_edges = get_line_graph_edges(child_rgraph_edges)
        # child_relation_hyperedge_nodes, child_relation_hyperedge_type = build_hypergraph_from_simple_graph(child_relation_nodes, child_relation_edges)
        
        # Generate child-to-parent mapping

        if len(child_hyperedge_nodes) > 2 and semantic_similarity_threshold > 0:
            child_hypergraph_nodes = set()
            for node_set in child_hyperedge_nodes:
                child_hypergraph_nodes |= set(node_set)
            cos_sim_hyperedges = build_cosine_similarity_hyperedges(sorted(list(child_hypergraph_nodes)), self.parent_graph_tokenizer, self.parent_graph_model, threshold=semantic_similarity_threshold, max_length=self.max_node_description_length)
            # print(cos_sim_hyperedges)
            # import sys
            # sys.exit(0)
            child_hyperedge_nodes += cos_sim_hyperedges
            child_hyperedge_types += ['self-loop'] * len(cos_sim_hyperedges)
            
        # child2parent_mapping = {}
        all_hypernodes = set()
        for node_set in child_hyperedge_nodes:
            all_hypernodes |= node_set
            # for node in node_set:
            #     parent = predict_cls_name(node, self.parent_graph_tokenizer, self.parent_graph_model, self.parent_graph_classifier, self.cls2name, self.predict_cache, self.max_node_description_length)
            #     child2parent_mapping[node] = parent
        child2parent_mapping = predict_category_name(all_hypernodes, self.parent_graph_tokenizer, self.parent_graph_model, self.parent_graph_classifier, cls2name=self.cls2name, predict_cache=self.predict_cache)
        # print(child2parent_mapping)
        # import sys
        # sys.exit(0)
        
        child_rgraph_edges = [(child2parent_mapping[src], child2parent_mapping[dst], {'type': self.relation2description[edge_type]}) for src, dst, edge_type in child_rgraph_edges]
        child_relation_nodes, child_relation_edges = get_line_graph_edges(child_rgraph_edges)
        child_relation_hyperedge_nodes, child_relation_hyperedge_type = build_hypergraph_from_simple_graph(child_relation_nodes, child_relation_edges)
        
        parent_hyperedge_nodes = []
        parent_nodes = child2parent_mapping.values()
        parent_hyperedge_index_inculded = set()
        for node in parent_nodes:
            parent_hyperedge_index_inculded |= self.parent_node2hyperedges[node]
        parent_nodes = set(parent_nodes)
        for index in parent_hyperedge_index_inculded:
            nodes = self.parent_hyperedge_nodes[index] & parent_nodes
            if len(nodes) > 1 and nodes not in parent_hyperedge_nodes:
                parent_hyperedge_nodes.append(nodes)
        
        # parent_hyperedge_nodes, parent_hyperedge_index = build_hyperedge_index_from_node_set(parent_hyperedge_nodes)
        # parent_hyperedge_nodes_encoded, parent_hyperedge_index = extend_hyperedge_index_with_tokenization(self.tokenizer, child_hyperedge_nodes, child_hyperedge_index)
        
        data = child_hyperedge_nodes[:]
        if parent_need:
            data += parent_hyperedge_nodes
        if relation_needed:
            data += child_relation_hyperedge_nodes
            # if parent_need:
            #     data += self.parent_relation_hyperedge_nodes
        
        result = self._format_hgraph_data(data)
        self.fol_cache[fol] = result
        self.changed = True
        return result
        
    def _format_hgraph_data(self, hyperedges, max_num_nodes=None, max_num_edges=None):
        if max_num_nodes is None:
            max_num_nodes = self.max_num_nodes
        if max_num_edges is None:
            max_num_edges = self.max_num_edges
            
        nodes_result = np.zeros(max_num_nodes)
        edges_result = np.zeros((2, max_num_edges))
        nodes, edges = build_hyperedge_index_from_node_set(hyperedges)
        nodes_encoded, hyperedge_index = extend_hyperedge_index_with_tokenization(self.tokenizer, nodes, edges)
        nodes_encoded_formated = nodes_encoded[:max_num_nodes]
        hyperedge_index_formated = hyperedge_index[:, hyperedge_index[0] < max_num_nodes][:, :max_num_edges]
        
        nodes_result[:nodes_encoded_formated.shape[0]] = nodes_encoded_formated
        edges_result[:, :hyperedge_index_formated.shape[1]] = hyperedge_index_formated
        return torch.tensor(edges_result.tolist(), dtype=torch.long), torch.tensor(nodes_result.tolist(), dtype=torch.long)
        
        