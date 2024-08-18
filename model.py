
import torch
import torch.nn as nn
import math
from layers import *
from torch.nn import Module, MultiheadAttention

class Intent_UniGCNII(Module):
    def __init__(self, opt, n_node):
        super().__init__()
        self.opt = opt
        self.disen = self.opt.disen
        self.n_node = n_node
        self.journal = self.opt.journal
        self.title = self.opt.title
        self.n_factor = opt.n_factor
        self.nhid = self.opt.nhid
        self.nhead = self.opt.nhead
        self.nlayer = self.opt.nlayer
        self.dim = self.opt.hiddenSize
        self.layer = int(self.opt.layer)
        self.embedding = nn.Embedding(n_node, self.dim)

        if self.disen:
            self.feat_latent_dim = self.dim // self.n_factor
            self.split_sections = [self.feat_latent_dim] * self.n_factor
            self.disenG = DisentangleGraph(opt=self.opt, dim=self.feat_latent_dim, alpha=self.opt.alpha, e=self.opt.e)
            self.intent_cition = nn.ModuleList([UniGCNII(self.opt, self.feat_latent_dim, self.nhid, self.nlayer, self.nhead) for i in range(self.n_factor)])
            if self.opt.journal and self.opt.title:
                self.fc = nn.Linear(self.opt.title_vec+self.opt.journal_vec, self.feat_latent_dim, bias=True)
                self.fc1 = nn.Linear(self.opt.title_vec+self.opt.journal_vec, self.dim, bias=True)
            elif self.opt.journal and not self.opt.title:
                self.fc = nn.Linear(self.opt.journal_vec, self.feat_latent_dim, bias=True)
                self.fc1 = nn.Linear(self.opt.journal_vec, self.dim, bias=True)
            elif not self.opt.journal and self.opt.title:
                self.fc = nn.Linear(self.opt.title_vec, self.feat_latent_dim, bias=True)
                self.fc1 = nn.Linear(self.opt.title_vec, self.dim, bias=True)

        else:
            self.feat_latent_dim = self.dim
            self.intent_cition = UniGCNII(self.opt, self.feat_latent_dim, self.nhid, self.nlayer, self.nhead)

            if self.opt.journal and self.opt.title:
                self.fc = nn.Linear(self.opt.title_vec+self.opt.journal_vec, self.dim, bias=True)
            elif self.opt.journal and not self.opt.title:
                self.fc = nn.Linear(self.opt.journal_vec, self.dim, bias=True)
            elif not self.opt.journal and self.opt.title:
                self.fc = nn.Linear(self.opt.title_vec, self.dim, bias=True)


        self.fc2 = nn.Linear(self.dim*2, self.dim, bias=True)
    
        self.w_k = 10
        self.fc4 = nn.Linear(2*self.dim, self.dim, bias=True)
        if opt.gcn:
            self.in_channels = self.opt.hiddenSize
            self.hidden_channels = self.opt.hidden_channels
            self.out_channels = self.opt.hiddenSize
            self.gcn = nn.ModuleList([GNN(self.in_channels, self.hidden_channels, self.out_channels) for _ in range(self.opt.num_layers)])

        encoder_transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.nhead, dim_feedforward=self.opt.hidden_dim,dropout=self.opt.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer,num_layers=1)

        self.leakyrelu = nn.LeakyReLU(self.opt.alpha)

        self.batch_size = self.opt.batch_size
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()
        if self.disen:
            # define for the additional losses
            self.classifier = nn.Linear(self.feat_latent_dim,  self.n_factor)
            self.loss_aux = nn.CrossEntropyLoss()
            self.intent_loss = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lr_dc_step, gamma=self.opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_disentangle_loss(self, intents_feat):
        # compute discrimination loss
        
        labels_row = [torch.ones(f.shape[0])*i for i, f in enumerate(intents_feat)] # (intent_num, (batch_size, latent_dim))
        labels = trans_to_cuda(torch.cat(tuple(labels_row), 0)).long()  # [2]
        intents_feat = torch.cat(tuple(intents_feat), 0) # 7*64

        pred = self.classifier(intents_feat)  # 7*7
        discrimination_loss = self.loss_aux(pred, labels)
        return discrimination_loss
    
    def compute_scores_int(self, citings, citing_seq_emb, item_embeddings, node_features,u_seqs_num):  # hidden(256,49,1368)
        if self.opt.integrate_model == 'transformer':
            out = self.transformer(citing_seq_emb)  # (batch_size, seq_len, dim)
        else: 
            out = citing_seq_emb

        citing_paper_vector = gather_indexes(out, u_seqs_num-1)

        citing_feats = node_features[citings]
        citing_vector = citing_paper_vector*citing_feats

        if self.disen:
            select_split = torch.split(citing_vector, self.split_sections, dim=-1)
            b = torch.split(item_embeddings[1:], self.split_sections, dim=-1)   
            score_all = []
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]
                citing_paper = sess_emb_int
                cited_paper = b[i]
                scores_int = torch.mm(citing_paper, torch.transpose(cited_paper, 1, 0))
                score_all.append(scores_int)
            score = torch.stack(score_all, dim=1)  
            scores = score.sum(1)   

        else:
            select = citing_paper_vector
            b = item_embeddings[1:]  
            scores = torch.matmul(select, b.transpose(1, 0))
        
        return scores
    
    def compute_scores_cont(self, citing, citing_papers_emb, item_embeddings, node_feats):  
        citing_feats = node_feats[citing]
        if self.opt.integrate_model == 'transformer':
        # transformer embedding
            citing_tran = self.transformer(citing_papers_emb) 
            citing_paper_vector = citing_tran.mean(dim=1)

        else:
            citing_paper_vector = citing_papers_emb.mean(dim=1)
        # recommendation
        citing_paper = torch.cat([citing_paper_vector, citing_feats], dim=1)
        cited_paper = torch.cat([item_embeddings[1:], node_feats], dim=1)

        scores = torch.matmul(citing_paper, cited_paper.transpose(1, 0))

        return scores
    
    def compute_scores_all(self, inputs, citings, item_emb_int, item_emb_cont, item_embeddings, node_features_cont,u_seqs_num):  # hidden(256,49,1368)
        node_emb = self.fc4(torch.cat([item_emb_int, item_emb_cont], dim=1))
        # node_emb = torch.cat([item_emb_int, item_emb_cont], dim=1)
        citing_seq_emb = node_emb[inputs]
        if self.opt.integrate_model == 'transformer':
            citing_tran = self.transformer(citing_seq_emb)  # (batch_size, seq_len, dim)
            out = citing_tran
        else: 
            out = citing_seq_emb  # (batch_size, dim)
        citing_paper_vector = gather_indexes(out, u_seqs_num-1)
        citing_feats = node_features_cont[citings]  # (batch_size, dim)

        if self.disen:
            select_split = torch.split(citing_paper_vector, self.split_sections, dim=-1)
            b = torch.split(item_embeddings[1:], self.split_sections, dim=-1)  

            score_all = []
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]
                citing_paper = torch.cat([sess_emb_int, citing_feats], dim=1)
                cited_paper = torch.cat([b[i], node_features_cont], dim=1) 
                scores_int = torch.mm(citing_paper, torch.transpose(cited_paper, 1, 0))

                score_all.append(scores_int)
            score = torch.stack(score_all, dim=1)  # (batch, n_intent, n_node)
            scores = score.sum(1)   # # (batch, n_node)

        else:
            select = citing_paper_vector
            b = item_embeddings[1:]  # n_nodes x latent_size
            scores = torch.matmul(select, b.transpose(1, 0))
        
        return scores


    def forward(self, citings, inputs, u_seqs,  u_seqs_num, H, H_content, node_features):

        node_features_cont = self.fc1(node_features)
        item_embeddings = self.embedding.weight
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embeddings = torch.cat([zeros, item_embeddings], 0)    # (14451,768)

        node_features_dis = item_embeddings[1:] * node_features_cont
        if self.disen:
            all_items = node_features_dis
            intents_cat = torch.mean(all_items, dim=0, keepdim=True)
            h_split = torch.split(all_items, self.split_sections, dim=-1)
            intent_split = torch.split(intents_cat, self.split_sections, dim=-1)   # 6*(1,128)
            
            h_ints = []
            intents_feat = []
            Hs = H
            for i in range(self.n_factor):
                h_int = h_split[i]
                int_emb = intent_split[i]

                Hs, vertex, edges, degV, degE = self.disenG(h_int, Hs, int_emb) 
                h_int = self.intent_cition[i](h_int, Hs, vertex, edges, degV, degE)   
                intent_p = int_emb.expand(self.n_node, int_emb.shape[-1]) 

                sim_val = h_int * intent_p
                cor_att = torch.sigmoid(sim_val)
                h_int = h_int * cor_att + h_int
                
                h_ints.append(h_int)   
                intents_feat.append(torch.mean(h_int, dim=0).unsqueeze(0))  
                
            h_stack = torch.stack(h_ints, dim=1)    
            dim_new = self.dim    # dim
            h_local = h_stack.reshape(self.n_node, dim_new)  
 
            self.intent_loss = self.compute_disentangle_loss(intents_feat)  

        else:

            h_local = self.intent_cition(item_embeddings)
                        
        item_emb_int = h_local
        citing_seq_emb = item_emb_int[u_seqs]
        scores = self.compute_scores_int(citings, citing_seq_emb, item_embeddings, node_features_cont, u_seqs_num)

        if self.opt.gcn:
            for layer in self.gcn:
                node_features_dis = layer(node_features_dis, H_content)
            item_emb_cont = node_features_dis

            scores = self.compute_scores_all(u_seqs, citings, item_emb_int, item_emb_cont, item_embeddings, node_features_cont,u_seqs_num)
            
                    
        return scores
    
def gather_indexes(output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
    
class GCN_rec(Module):
    def __init__(self, opt, num_node):
        super(GCN_rec, self).__init__()
        # HYPER PARA
        self.opt = opt 
        self.num_node = num_node
        self.dim = self.opt.hiddenSize
        self.embedding = nn.Embedding(self.num_node, self.dim)

        self.in_channels = self.opt.hiddenSize
        self.hidden_channels = self.opt.hidden_channels
        self.out_channels = self.opt.hiddenSize
        self.gcn = nn.ModuleList([GNN(self.in_channels, self.hidden_channels, self.out_channels) for _ in range(self.opt.num_layers)])


        if self.opt.journal and self.opt.title:
            self.f1 = nn.Linear(self.opt.title_vec+self.opt.journal_vec, self.dim, bias=True)
        elif self.opt.journal and not self.opt.title:
            self.f1 = nn.Linear(self.opt.journal_vec, self.dim, bias=True)
        elif not self.opt.journal and self.opt.title:
            self.f1 = nn.Linear(self.opt.title_vec, self.dim, bias=True)

        self.batch_size = self.opt.batch_size
        self.nhead = self.opt.nhead
        encoder_transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.nhead, dim_feedforward=self.opt.hidden_dim,dropout=self.opt.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer,num_layers=1)
        self.intent_loss = 0
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) # 优化器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)  # 学习率衰减计划
        self.reset_parameters() # 初始化参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, citing, h_items, u_seqs, u_seqs_num, adj_mat, H_content, node_feature):
        node_feature = self.f1(node_feature) 
        item_embeddings = self.embedding.weight
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embed = torch.cat([zeros, item_embeddings], 0)   
        item_embeddings_new = item_embed[1:] * node_feature
        for layer in self.gcn:
            item_embeddings_new = layer(item_embeddings_new, adj_mat)
        item_emb = item_embeddings_new
        # item_emb = self.gcn(item_embeddings_new, adj_mat)
        citing_papers_emb = item_emb[u_seqs]
        scores = self.compute_scores(citing, citing_papers_emb, item_embeddings_new, node_feature)
        return scores
    
    def compute_scores(self, citing, citing_papers_emb, item_embeddings, node_feature):  # hidden(256,49,1368)
        citing_feats = node_feature[citing]
        if self.opt.integrate_model == 'transformer':
        # transformer embedding
            citing_tran = self.transformer(citing_papers_emb)  # (batch_size, seq_len, dim)
            citing_paper_vector = citing_tran.mean(dim=1)

        else:
            citing_paper_vector = citing_papers_emb.mean(dim=1)
        # recommendation
        citing_paper = torch.cat([citing_paper_vector, citing_feats], dim=1)
        cited_paper = torch.cat([item_embeddings, node_feature], dim=1)

        scores = torch.matmul(citing_paper, cited_paper.transpose(1, 0))

        return scores

