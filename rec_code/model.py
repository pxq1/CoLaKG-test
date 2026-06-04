import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
    
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKG(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset, 
                 adj_matrix=None, 
                 semantic_emb=None, 
                 user_semantic_emb=None,):
        super(CoLaKG, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
   
        self.user_semantic_emb = user_semantic_emb.to(world.device)
        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.use_fusion_gate = bool(self.config.get('fusion_gate', 0))
        self.gate_type = self.config.get('gate_type', 'scalar')
        self.prop_norm = bool(self.config.get('prop_norm', 0))
        self.graph_gamma = self.config.get('graph_gamma', -1.0)
        self.use_social = bool(self.config.get('use_social', 0))
        self.social_alpha = self.config.get('social_alpha', 0.1)
        self.semantic_score_alpha = self.config.get('semantic_score_alpha', 0.0)
        self.raw_semantic_score_alpha = self.config.get('raw_semantic_score_alpha', 0.0)
        self.semantic_cl_weight = self.config.get('semantic_cl_weight', 0.0)
        self.semantic_cl_tau = self.config.get('semantic_cl_tau', 0.2)
        self.pop_score_alpha = self.config.get('pop_score_alpha', 0.0)
        self.neighbor_score_alpha = self.config.get('neighbor_score_alpha', 0.0)
        self.neighbor_score_steps = max(int(self.config.get('neighbor_score_steps', 1)), 1)
        self.neighbor_score_mutual = bool(self.config.get('neighbor_score_mutual', 0))
        self.neighbor_train_alpha = self.config.get('neighbor_train_alpha', 0.0)
        self.cf_score_alpha = self.config.get('cf_score_alpha', 0.0)
        self.cf_neighbor_k = max(int(self.config.get('cf_neighbor_k', 20)), 1)
        self.simgcl_weight = self.config.get('simgcl_weight', 0.0)
        self.simgcl_tau = self.config.get('simgcl_tau', 0.2)
        self.simgcl_eps = self.config.get('simgcl_eps', 0.1)
        self.simgcl_start_epoch = self.config.get('simgcl_start_epoch', 0)
        self.simgcl_stop_epoch = self.config.get('simgcl_stop_epoch', -1)
        self.loss_type = self.config.get('loss_type', 'bpr')
        self.softmax_weight = self.config.get('softmax_weight', 1.0)
        self.softmax_tau = self.config.get('softmax_tau', 1.0)
        self.softmax_mask_pos = bool(self.config.get('softmax_mask_pos', 1))
        self.softmax_label_smoothing = self.config.get('softmax_label_smoothing', 0.0)
        self.softmax_start_epoch = self.config.get('softmax_start_epoch', 0)
        self.softmax_stop_epoch = self.config.get('softmax_stop_epoch', -1)
        self.current_epoch = 0
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
   
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.SocialGraph = self.dataset.getSocialGraph() if self.use_social and hasattr(self.dataset, 'getSocialGraph') else None
        self.semantic_map = nn.Linear(1024, self.latent_dim)
        self.user_semantic_map = nn.Linear(1024, self.latent_dim)
        item_popularity = np.bincount(self.dataset.trainItem, minlength=self.num_items).astype(np.float32)
        item_popularity = np.log1p(item_popularity)
        item_popularity = (item_popularity - item_popularity.mean()) / (item_popularity.std() + 1e-8)
        self.register_buffer('item_popularity_prior', torch.from_numpy(item_popularity))
        train_pos_mask = torch.zeros((self.num_users, self.num_items), dtype=torch.bool)
        for user, positives in enumerate(self.dataset.allPos):
            if len(positives) > 0:
                train_pos_mask[user, torch.as_tensor(positives, dtype=torch.long)] = True
        self.register_buffer('train_pos_mask', train_pos_mask)
        self.register_buffer('semantic_mutual_mask', self.build_semantic_mutual_mask(), persistent=False)
        self.register_buffer('cf_adj_matrix', self.build_cf_adj_matrix(), persistent=False)
        print(f"lgn is already to go(drop_edge:{self.config['use_drop_edge']})")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.W_u = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W_u.data, gain=1.414)
        self.a_u = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a_u.data, gain=1.414)
        gate_shape = (self.latent_dim,) if self.gate_type == 'vector' else (1,)
        self.item_semantic_gate = nn.Parameter(torch.zeros(gate_shape))
        self.user_semantic_gate = nn.Parameter(torch.zeros(gate_shape))
        self.neighbor_fusion_gate = nn.Parameter(torch.zeros(gate_shape))
        self.alpha=0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # print("save_txt")

    def build_semantic_mutual_mask(self):
        neighbor_sets = [set(row.tolist()) for row in self.adj_matrix.cpu()]
        mutual_mask = torch.zeros_like(self.adj_matrix, dtype=torch.float32)
        for item_id, neighbors in enumerate(self.adj_matrix.cpu().tolist()):
            for offset, neighbor_id in enumerate(neighbors):
                if item_id in neighbor_sets[neighbor_id]:
                    mutual_mask[item_id, offset] = 1.0
        return mutual_mask

    def aggregate_neighbor_scores(self, base_scores, neighbor_index, mutual_mask=None):
        selected_scores = base_scores[:, neighbor_index]
        if mutual_mask is None:
            return torch.mean(selected_scores, dim=2)
        mask = mutual_mask.to(selected_scores.device).unsqueeze(0)
        denom = torch.sum(mask, dim=2).clamp_min(1.0)
        masked_scores = torch.sum(selected_scores * mask, dim=2) / denom
        plain_scores = torch.mean(selected_scores, dim=2)
        has_mutual = torch.sum(mask, dim=2) > 0
        return torch.where(has_mutual, masked_scores, plain_scores)

    def build_cf_adj_matrix(self):
        if self.cf_score_alpha == 0 or not hasattr(self.dataset, 'trainUser') or not hasattr(self.dataset, 'trainItem'):
            return torch.zeros((self.num_items, 1), dtype=torch.long)
        k = min(self.cf_neighbor_k, max(self.num_items - 1, 1))
        item_user = torch.zeros((self.num_items, self.num_users), dtype=torch.float32)
        item_ids = torch.as_tensor(self.dataset.trainItem, dtype=torch.long)
        user_ids = torch.as_tensor(self.dataset.trainUser, dtype=torch.long)
        item_user[item_ids, user_ids] = 1.0
        item_user = F.normalize(item_user, p=2, dim=1)
        item_sim = torch.matmul(item_user, item_user.t())
        item_sim.fill_diagonal_(-1.0)
        return torch.topk(item_sim, k=k, dim=1).indices.long()

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self, perturbed=False):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        if self.use_fusion_gate:
            item_semantic_weight = torch.sigmoid(self.item_semantic_gate)
            items_emb_merged = (1 - item_semantic_weight) * items_emb + item_semantic_weight * items_semantic_emb
        else:
            items_emb_merged = (items_emb + items_semantic_emb) / 2
        
        user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
        user_semantic_emb = self.user_semantic_map(user_semantic_emb)
        user_semantic_emb = F.elu(user_semantic_emb)
        user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
        if self.use_fusion_gate:
            user_semantic_weight = torch.sigmoid(self.user_semantic_gate)
            users_emb_merged = (1 - user_semantic_weight) * users_emb + user_semantic_weight * user_semantic_emb
        else:
            users_emb_merged = (users_emb + user_semantic_emb) / 2

        if self.SocialGraph is not None and self.social_alpha > 0:
            social_users_emb = torch.sparse.mm(self.SocialGraph, users_emb_merged)
            users_emb_merged = (1 - self.social_alpha) * users_emb_merged + self.social_alpha * social_users_emb
        
        
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]  # N,L,d1

        # x = self.attentions(neighbor_semantic_emb, neighbor_emb, items_semantic_emb0)
        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0
        
        Wh = torch.matmul(h, self.W)  # N,L,d
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1],1)  # N,L,d1
        Wh0 = torch.matmul(h0, self.W)  # N,L,d
        
        W_concat = torch.cat((Wh, Wh0), dim=-1) # N,L,2d
        
        attention = torch.matmul(W_concat, self.a).squeeze(-1) # N,L
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1) # N,L
    
        attention = F.dropout(attention, self.dropout_neighbor, training=self.training) # N,L
        attention = attention.unsqueeze(-1)
     
        h_prime = attention * value_emb

        h_prime = torch.sum(h_prime, dim=1)
        
        h_prime = F.elu(h_prime)
      

        if self.use_fusion_gate:
            neighbor_weight = torch.sigmoid(self.neighbor_fusion_gate)
            items_emb_merged = (1 - neighbor_weight) * items_emb_merged + neighbor_weight * h_prime
        else:
            items_emb_merged = (items_emb_merged + h_prime ) / 2
        
        # items_emb = F.elu(items_emb)
       
        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]
        
        if self.config['use_drop_edge']:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.prop_norm:
                all_emb = F.normalize(all_emb, p=2, dim=1)
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed and self.simgcl_eps > 0:
                random_noise = torch.rand_like(all_emb)
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, p=2, dim=1) * self.simgcl_eps
            embs.append(all_emb)
        if self.graph_gamma >= 0 and len(embs) > 1:
            prop_emb = torch.mean(torch.stack(embs[1:], dim=1), dim=1)
            light_out = self.graph_gamma * embs[0] + (1 - self.graph_gamma) * prop_emb
        else:
            embs = torch.stack(embs, dim=1)
            #print(embs.size())
            light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def score_all_items(self, users, users_emb, items_emb):
        scores = torch.matmul(users_emb, items_emb.t())
        if self.neighbor_score_alpha != 0:
            neighbor_scores = scores
            mutual_mask = self.semantic_mutual_mask if self.neighbor_score_mutual else None
            for _ in range(self.neighbor_score_steps):
                neighbor_scores = self.aggregate_neighbor_scores(neighbor_scores, self.adj_matrix, mutual_mask)
            scores = scores + self.neighbor_score_alpha * neighbor_scores
        if self.cf_score_alpha != 0:
            cf_neighbor_scores = torch.mean(scores[:, self.cf_adj_matrix], dim=2)
            scores = scores + self.cf_score_alpha * cf_neighbor_scores
        if self.semantic_score_alpha > 0:
            semantic_users = F.normalize(F.elu(self.user_semantic_map(self.user_semantic_emb))[users.long()], p=2, dim=1)
            semantic_items = F.normalize(F.elu(self.semantic_map(self.semantic_emb)), p=2, dim=1)
            scores = scores + self.semantic_score_alpha * torch.matmul(semantic_users, semantic_items.t())
        if self.raw_semantic_score_alpha != 0:
            raw_semantic_users = F.normalize(self.user_semantic_emb[users.long()], p=2, dim=1)
            raw_semantic_items = F.normalize(self.semantic_emb, p=2, dim=1)
            scores = scores + self.raw_semantic_score_alpha * torch.matmul(raw_semantic_users, raw_semantic_items.t())
        if self.pop_score_alpha != 0:
            scores = scores + self.pop_score_alpha * self.item_popularity_prior.unsqueeze(0)
        return scores

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        scores = self.score_all_items(users.long(), users_emb, items_emb)
        rating = self.f(scores)
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]
        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb_ego0, all_items

    def semantic_cl_loss(self, rec_emb, semantic_emb):
        rec_emb = F.normalize(rec_emb, p=2, dim=1)
        semantic_emb = F.normalize(F.elu(semantic_emb), p=2, dim=1)
        logits = torch.matmul(rec_emb, semantic_emb.t()) / self.semantic_cl_tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def simgcl_cl_loss(self, view1, view2, ids):
        ids = torch.unique(ids.long())
        view1 = F.normalize(view1[ids], p=2, dim=1)
        view2 = F.normalize(view2[ids], p=2, dim=1)
        logits = torch.matmul(view1, view2.t()) / self.simgcl_tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def get_simgcl_weight(self):
        if self.simgcl_weight <= 0:
            return 0.0
        epoch = getattr(self, 'current_epoch', 0)
        if epoch < self.simgcl_start_epoch:
            return 0.0
        if self.simgcl_stop_epoch >= 0 and epoch >= self.simgcl_stop_epoch:
            return 0.0
        return self.simgcl_weight

    def get_softmax_weight(self):
        if self.loss_type not in ('softmax', 'bpr_softmax'):
            return 0.0
        epoch = getattr(self, 'current_epoch', 0)
        if epoch < self.softmax_start_epoch:
            return 0.0
        if self.softmax_stop_epoch >= 0 and epoch >= self.softmax_stop_epoch:
            return 0.0
        if self.loss_type == 'softmax':
            return 1.0
        return self.softmax_weight

    def full_item_softmax_loss(self, users, pos, users_emb, all_items):
        tau = max(float(self.softmax_tau), 1e-8)
        scores = self.score_all_items(users.long(), users_emb, all_items) / tau
        labels = pos.long()
        if self.softmax_mask_pos:
            row_ids = torch.arange(scores.size(0), device=scores.device)
            positive_mask = self.train_pos_mask[users.long()].clone()
            positive_mask[row_ids, labels] = False
            scores = scores.masked_fill(positive_mask, -1e9)
        return F.cross_entropy(scores, labels, label_smoothing=self.softmax_label_smoothing)
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0, users_emb_ego0, all_items) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2) + 
                         pos_emb_ego0.norm(2).pow(2) + 
                         neg_emb_ego0.norm(2).pow(2) + 
                         users_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        if self.neighbor_train_alpha != 0:
            pos_neighbor_emb = torch.mean(all_items[self.adj_matrix[pos.long()]], dim=1)
            neg_neighbor_emb = torch.mean(all_items[self.adj_matrix[neg.long()]], dim=1)
            pos_scores = pos_scores + self.neighbor_train_alpha * torch.sum(users_emb * pos_neighbor_emb, dim=1)
            neg_scores = neg_scores + self.neighbor_train_alpha * torch.sum(users_emb * neg_neighbor_emb, dim=1)
        if self.semantic_score_alpha > 0:
            semantic_users = F.normalize(F.elu(users_emb_ego0), p=2, dim=1)
            semantic_pos = F.normalize(F.elu(pos_emb_ego0), p=2, dim=1)
            semantic_neg = F.normalize(F.elu(neg_emb_ego0), p=2, dim=1)
            pos_scores = pos_scores + self.semantic_score_alpha * torch.sum(semantic_users * semantic_pos, dim=1)
            neg_scores = neg_scores + self.semantic_score_alpha * torch.sum(semantic_users * semantic_neg, dim=1)
        if self.raw_semantic_score_alpha != 0:
            raw_semantic_users = F.normalize(self.user_semantic_emb[users.long()], p=2, dim=1)
            raw_semantic_pos = F.normalize(self.semantic_emb[pos.long()], p=2, dim=1)
            raw_semantic_neg = F.normalize(self.semantic_emb[neg.long()], p=2, dim=1)
            pos_scores = pos_scores + self.raw_semantic_score_alpha * torch.sum(raw_semantic_users * raw_semantic_pos, dim=1)
            neg_scores = neg_scores + self.raw_semantic_score_alpha * torch.sum(raw_semantic_users * raw_semantic_neg, dim=1)
        if self.pop_score_alpha != 0:
            pos_scores = pos_scores + self.pop_score_alpha * self.item_popularity_prior[pos.long()]
            neg_scores = neg_scores + self.pop_score_alpha * self.item_popularity_prior[neg.long()]
        
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        active_softmax_weight = self.get_softmax_weight()
        loss = bpr_loss
        if self.loss_type == 'softmax' and active_softmax_weight > 0:
            loss = torch.zeros_like(bpr_loss)
        if active_softmax_weight > 0:
            softmax_loss = self.full_item_softmax_loss(users, pos, users_emb, all_items)
            loss = loss + active_softmax_weight * softmax_loss
        if self.semantic_cl_weight > 0:
            cl_loss = self.semantic_cl_loss(users_emb, users_emb_ego0)
            cl_loss = cl_loss + self.semantic_cl_loss(pos_emb, pos_emb_ego0)
            loss = loss + self.semantic_cl_weight * cl_loss
        active_simgcl_weight = self.get_simgcl_weight()
        if active_simgcl_weight > 0:
            users_view1, items_view1 = self.computer(perturbed=True)
            users_view2, items_view2 = self.computer(perturbed=True)
            cl_loss = self.simgcl_cl_loss(users_view1, users_view2, users)
            cl_loss = cl_loss + self.simgcl_cl_loss(items_view1, items_view2, pos)
            loss = loss + active_simgcl_weight * cl_loss
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
