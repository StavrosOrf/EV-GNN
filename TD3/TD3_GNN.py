
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, TAGConv
from torch_geometric.nn import global_add_pool, SAGPooling, ASAPooling, GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected

import torch_geometric.transforms as T

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class GNN_Redisual_Feature_Extractor(nn.Module):
    def __init__(self,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 num_heads=2,
                 device=torch.device('cpu')):

        super(GNN_Redisual_Feature_Extractor, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(6, feature_dim)
        self.cs_embedding = nn.Linear(4, feature_dim)
        self.tr_embedding = nn.Linear(2, feature_dim)
        self.env_embedding = nn.Linear(5, feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        # GCN layer with 8 input, 16 output
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim)
        # self.gcn_conv3 = GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim)
        # self.gcn_conv4 = GCNConv(3*GNN_hidden_dim, 2*GNN_hidden_dim)
        self.gcn_conv5 = GCNConv(2*GNN_hidden_dim, GNN_hidden_dim)

    def forward(self, data):

        if isinstance(data.env_features, np.ndarray):
            ev_features = torch.from_numpy(
                data.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(
                data.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(
                data.tr_features).float().to(self.device)
            env_features = torch.from_numpy(
                data.env_features).float().to(self.device)
            edge_index = torch.from_numpy(
                data.edge_index).long().to(self.device)
        else:
            ev_features = data.ev_features
            cs_features = data.cs_features
            tr_features = data.tr_features
            env_features = data.env_features
            edge_index = data.edge_index
            node_types = data.node_types

        # edge_index = to_undirected(edge_index)

        sample_node_length = data.sample_node_length

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(
            total_nodes, self.feature_dim, device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(data.ev_indexes) != 0:
            embedded_x[data.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[data.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[data.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[data.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)

        # Apply GCN and GAT layers with the unified edge index
        # print(f'Embedded Node Features shape: {embedded_x.shape}')
        x1 = self.gcn_conv(embedded_x, edge_index)
        x = F.relu(x1)

        x2 = self.gcn_conv2(x, edge_index)
        x = F.relu(x2)

        # x3 = self.gcn_conv3(x, edge_index)
        # x = F.relu(x3)

        # x4 = self.gcn_conv4(x, edge_index)
        # x = F.relu(x4)

        # x = x + x2
        x = self.gcn_conv5(x, edge_index)
        x = F.relu(x)

        x = x + x1

        # make batch sample mask
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        counter = 0
        for i in range(len(sample_node_length)):
            batch[counter: counter + sample_node_length[i]] = i
            counter += sample_node_length[i]

        pooled_embedding = global_mean_pool(x, batch=batch)

        # print(f'Pooled embedding shape: {pooled_embedding.shape}')
        return pooled_embedding


class GNN_Feature_Extractor(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 num_heads=2,
                 device=torch.device('cpu')):

        super(GNN_Feature_Extractor, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        # GCN layer with 8 input, 16 output
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim)
        self.gcn_conv3 = GCNConv(2*GNN_hidden_dim, 4*GNN_hidden_dim)

        # self.gcn_conv = GATv2Conv(feature_dim, GNN_hidden_dim, heads=num_heads)
        # self.gcn_conv2 = GATv2Conv(
        #     num_heads*GNN_hidden_dim, GNN_hidden_dim, heads=num_heads)
        # self.gcn_conv3 = GATv2Conv(
        #     num_heads*GNN_hidden_dim, 2*GNN_hidden_dim, heads=num_heads)

        # GAT with multiple attention heads
        # self.gat_conv = GATConv(GNN_hidden_dim, 64, heads=num_heads)

        # Global pooling with attention for a consistent graph embedding
        # self.attn_pool = AttentionalAggregation(
        #     gate_nn=nn.Sequential(
        #         nn.Linear(4*GNN_hidden_dim, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 1),  # Attention scores
        #     )
        # )
        # self.sag_pool = ASAPooling(4*GNN_hidden_dim)
        # self.sag_pool = SAGPooling(4*GNN_hidden_dim, min_score=0.35)

        # self.attn_pool = TopKPooling(128, ratio=0.5)
        # self.attn_pool = TopKPooling(128, ratio=0.5)

        # self.attn_pool = TopKPooling(128, ratio=0.5)

    def forward(self, data):

        if isinstance(data.env_features, np.ndarray):
            ev_features = torch.from_numpy(
                data.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(
                data.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(
                data.tr_features).float().to(self.device)
            env_features = torch.from_numpy(
                data.env_features).float().to(self.device)
            edge_index = torch.from_numpy(
                data.edge_index).long().to(self.device)

        else:
            ev_features = data.ev_features
            cs_features = data.cs_features
            tr_features = data.tr_features
            env_features = data.env_features
            edge_index = data.edge_index
        # edge_index = to_undirected(edge_index)

        sample_node_length = data.sample_node_length

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(
            total_nodes, self.feature_dim, device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(data.ev_indexes) != 0:
            embedded_x[data.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[data.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[data.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[data.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)

        # Apply GCN and GAT layers with the unified edge index
        # print(f'Embedded Node Features shape: {embedded_x.shape}')
        x = self.gcn_conv(embedded_x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv3(x, edge_index)
        # x = self.gat_conv(x, edge_index)
        x = F.relu(x)
        # print(f'Last GNN layer shape: {x.shape}')

        # for each graph get the graph embedding

        # node_counter = 0
        # pooled_embedding = torch.zeros(
        #     (len(sample_node_length), x.shape[1]), device=self.device)

        # make batch sample mask
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        counter = 0
        for i in range(len(sample_node_length)):
            batch[counter: counter + sample_node_length[i]] = i
            counter += sample_node_length[i]

        # for i in range(len(sample_node_length)):
        #     # Apply global attention-based pooling
        #     # print(f'node_counter+sample_node_length[i]: {node_counter+sample_node_length[i]}')
        #     # print(x[node_counter:node_counter+sample_node_length[i],:].shape)

        #     # pooled_embedding[i, :] = self.attn_pool(
        #     #     x[node_counter:node_counter+sample_node_length[i], :], None) #Global Attention Pooling

        #     # print(f'graph {i} pooling')
        #     # print(x[node_counter:node_counter+sample_node_length[i], :].shape)
        #     # print(global_mean_pool(
        #     #     x[node_counter:node_counter+sample_node_length[i], :],batch=None))
        #     # pooled_embedding[i, :] = global_mean_pool(
        #     #     x[node_counter:node_counter+sample_node_length[i], :],batch=None)

        #     pooled_embedding[i, :] = global_max_pool(
        #         x[node_counter:node_counter+sample_node_length[i], :], batch=None)

        #     node_counter += sample_node_length[i]

        # pooled_embedding = global_add_pool(x, batch=batch)
        # pooled_embedding = global_mean_pool(x, batch=batch)
        # print(f'batch: {batch}')
        # print(f'x shape: {x.shape}')
        # print(f'{self.sag_pool(x, edge_index=edge_index, batch=batch)}')
        # x, edge_index, _, batch, _, _ = self.sag_pool(
        #     x, edge_index=edge_index, batch=batch)
        # print(f'x~ shape: {x.shape}')
        pooled_embedding = global_mean_pool(x, batch=batch)

        # print(f'Pooled embedding: {pooled_embedding}')
        # print(f'Pooled embedding shape: {pooled_embedding.shape}')

        return pooled_embedding


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, fx_node_sizes,
                 fx_dim=8, fx_GNN_hidden_dim=32, num_heads=2,
                 mlp_hidden_dim=256):
        super(Actor, self).__init__()

        # self.feature_extractor = feature_extractor
        self.feature_extractor = GNN_Feature_Extractor(feature_dim=fx_dim,
                                                       fx_node_sizes=fx_node_sizes,
                                                       GNN_hidden_dim=fx_GNN_hidden_dim,
                                                       num_heads=num_heads,
                                                       device=torch.device('cuda'))

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # print(f'Actor input: {state}')
        state = self.feature_extractor(state)
        # print(f'Actor input: {state.shape}')
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.sigmoid(self.l3(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 fx_node_sizes, fx_dim=8, fx_GNN_hidden_dim=32, num_heads=2, mlp_hidden_dim=256):
        super(Critic, self).__init__()

        # self.feature_extractor = feature_extractor
        self.feature_extractor = GNN_Feature_Extractor(feature_dim=fx_dim,
                                                       fx_node_sizes=fx_node_sizes,
                                                       GNN_hidden_dim=fx_GNN_hidden_dim,
                                                       num_heads=num_heads,
                                                       device=torch.device('cuda'))

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, mlp_hidden_dim)
        self.l5 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l6 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state, action):

        state = self.feature_extractor(state)

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.feature_extractor(state)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_GNN(object):
    def __init__(
            self,
            action_dim,
            max_action,
            fx_node_sizes,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            fx_dim=8,
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=256,
            fx_num_heads=2,
            lr=3e-4,
            **kwargs
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        ###### Check this   ########
        gnn_output_dim = 4*fx_GNN_hidden_dim
        #################################

        self.actor = Actor(gnn_output_dim,
                           action_dim,
                           max_action,
                           fx_node_sizes=fx_node_sizes,
                           fx_dim=fx_dim,
                           fx_GNN_hidden_dim=fx_GNN_hidden_dim,
                           mlp_hidden_dim=mlp_hidden_dim,
                           ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(gnn_output_dim,
                             action_dim,
                             fx_node_sizes=fx_node_sizes,
                             fx_dim=fx_dim,
                             fx_GNN_hidden_dim=fx_GNN_hidden_dim,
                             mlp_hidden_dim=mlp_hidden_dim,
                             ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, **kwargs):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = state.to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            return critic_loss.item(), actor_loss.item()

        return None, None

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
