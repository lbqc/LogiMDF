import torch.nn as nn
from torch_geometric.nn import HypergraphConv

from layers.opera_best_bert import evaluate_expression
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric import data,loader
from .hypergraph import HypergraphLearning


class HGCN(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num, num_nodes=3, num_hyperedges=3):
        super(HGCN, self).__init__()
        self.gcn_layers = nn.ModuleList(
            [HypergraphConv(in_channels, out_channels) for _ in range(layer_num)]
        )
        # self.gcn_learning_layers = nn.ModuleList(
        #     [HypergraphLearning(out_channels, num_nodes, num_hyperedges) for _ in range(layer_num)]
        # )
        self.activate = nn.LeakyReLU(0.2)
        self.layer_num = layer_num
        
    def forward(self, x, edge_index):

        # for gcn, gcn_learing in zip(self.gcn_layers, self.gcn_learning_layers):
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = self.activate(x)
            # x = gcn_learing(x)
        return x
    
    
class BertFolHGraph(nn.Module):
    def __init__(self, bert, opt, tokenizer, *args, **kwargs):
        super(BertFolHGraph, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.llm_name = opt.llm_name.lower()
        if 'ans' in opt.llm_name.lower():
            self.dense1 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
            self.dense2 = nn.Linear(self.bert.config.hidden_size*1, self.bert.config.hidden_size)
            print(opt.llm_name)
        elif 'fusion' in opt.llm_name.lower():
            self.dense1 = nn.Linear(self.bert.config.hidden_size*5, self.bert.config.hidden_size)
            self.dense2 = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size)
            print(opt.llm_name)
        else:
            self.dense1 = nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size)
            self.dense2 = nn.Linear(self.bert.config.hidden_size*3, self.bert.config.hidden_size)
        # self.dense = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.graph_dense11 = nn.Linear(opt.nodes_num, 1)
        self.graph_dense12 = nn.Linear(opt.nodes_num, 1)
        self.graph_dense13 = nn.Linear(opt.nodes_num, 1)
        self.graph_dense14 = nn.Linear(opt.nodes_num, 1)
        # self.graph_dense2 = nn.Linear(self.bert.config.hidden_size, opt.polarities_dim)
        self.gcn = HGCN(self.bert.config.hidden_size, self.bert.config.hidden_size, opt.gcn_num, opt.nodes_num, opt.hyperedges_num)
        self.tokenizer = tokenizer
        self.with_text = opt.with_text
        self.batch_size = opt.batch_size
        
        
    def forward(self, inputs):
        fol_bert_indices, fol_bert_type,fol_bert_mask,ti,tt,tm,mlm,graph_indexs,graph_texts = inputs
        graph_indexs = graph_indexs.transpose(0,1)
        graph_texts = graph_texts.transpose(0,1)
        try:
            assert graph_texts.size(0) == 4 == graph_indexs.size(0)
        except:
            print(graph_texts.size(0), graph_indexs.size(0))
            assert graph_texts.size(0) == 4 == graph_indexs.size(0)
        for _index,(graph_index,graph_text) in enumerate(zip(graph_indexs,graph_texts)):
            if 'ans' in self.llm_name and _index != int(self.llm_name[-1]) - 1:
                continue
            if 'fusion' not in self.llm_name and _index == 3:
                continue
            graph_hidden_state,_= self.bert(graph_text,return_dict=False)
            # print(graph_hidden_state.size(),graph_index.size(),graph_type.size())
            # for sgraph_hidden_state,sgraph_index,sgraph_type in zip(graph_hidden_state,graph_index,graph_type):
            #     print(sgraph_hidden_state.size(), sgraph_index[:,:len(sgraph_type[sgraph_type >= 0])].size(), sgraph_type[sgraph_type >= 0].size())
            
            data_list = [
                Data(x=sgraph_hidden_state, edge_index=sgraph_index) for sgraph_hidden_state,sgraph_index in zip(graph_hidden_state,graph_index)
            ]
            batch_data = list(loader.DataLoader(data_list, batch_size=self.batch_size,shuffle=False))[0]
            # print(batch_data.x.size(), batch_data.edge_index.size(), batch_data.edge_type.size())
            graph_text_out = self.gcn(batch_data.x, batch_data.edge_index)
            
            # x_batch = torch.cat([single_graph_text for single_graph_text in graph_hidden_state],dim=0)
            # # print(graph_index.size(),graph_type.size())
            # # for single_graph_index,single_graph_type in zip(graph_index,graph_type):
            # #     print(single_graph_index.size(),single_graph_type.size())
            #     # print(single_graph_index[single_graph_type>=0])
            # edge_index_batch = torch.cat([single_graph_index[single_graph_type>=0] for single_graph_index,single_graph_type in zip(graph_index,graph_type)],dim=0)
            # edge_type_batch = torch.cat([single_graph_type[single_graph_type>=0] for single_graph_type in graph_type],dim=0)
            # gbatch = torch.cat([torch.full_like(single_graph_text[:, 0], i) for i, single_graph_text in enumerate(graph_hidden_state)],dim=0)
            # print(edge_type_batch.size(),edge_index_batch.size(),x_batch.size(),gbatch.size())
            # graph_text_out = self.gcn(x_batch, edge_index_batch, edge_type_batch, gbatch)
            graph_output_split = torch.split(graph_text_out, [len(d) for d in graph_hidden_state], dim=0)
            graph_text_out = torch.cat([_item.unsqueeze(0) for _item in graph_output_split],0)

            # graph_text_out = self.gcn(graph_text_out, edge_index_batch, edge_type_batch, batch)
            if _index == 0 or 'ans' in self.llm_name:
                _graph_out = self.graph_dense11(graph_text_out.transpose(1,2)).squeeze(-1)
            elif _index == 1:
                _graph_out = self.graph_dense12(graph_text_out.transpose(1,2)).squeeze(-1)
            elif _index == 2:
                _graph_out = self.graph_dense13(graph_text_out.transpose(1,2)).squeeze(-1)
            elif _index == 3:
                _graph_out = self.graph_dense14(graph_text_out.transpose(1,2)).squeeze(-1)
            
            if _index == 0 or 'ans' in self.llm_name:
                graph_out = _graph_out
            else:
                graph_out = torch.concat((graph_out,_graph_out),-1)
                
            # print(_index)
            # print(self.llm_name)
            # print(graph_out.shape)
        
        # hidden_state,pooled_output= self.bert(fol_bert_indices, token_type_ids=fol_bert_type,attention_mask=fol_bert_mask,return_dict=False)
        if self.with_text:
            text_hidden_state,pooled_output= self.bert(ti, token_type_ids=tt,attention_mask=tm,return_dict=False)
            text_hidden_state = text_hidden_state[mlm >= 0].view(text_hidden_state.size(0),1,text_hidden_state.size(-1)).squeeze(1)

        # out = torch.zeros([hidden_state.size(0),hidden_state.size(-1)]).cuda()
        # for index in range(hidden_state.size(0)):
            # out[index] = evaluate_expression(fol_bert_indices[index],hidden_state[index]).squeeze()

        
        if self.with_text:
            out = torch.cat((graph_out,text_hidden_state),-1)
            out = self.dropout(out)
            out = self.dense1(out)
        else:
            out = self.dropout(graph_out)
            out = self.dense2(out)

        self.tokenizer.get_labels()
        logits = [
            torch.mm(
                out[:,:],
                self.bert.embeddings.word_embeddings.weight[i].transpose(1,0)
            ) for i in self.tokenizer.prompt_label_idx
        ]
        # logits = self.dense(out)
        # print(logits.size(), logits.device)
        return logits