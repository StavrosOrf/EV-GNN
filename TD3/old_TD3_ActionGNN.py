
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, TAGConv
from torch_geometric.nn import global_add_pool, SAGPooling, ASAPooling
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected, softmax

import torch_geometric.transforms as T

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self,
                 max_action,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 discrete_actions=1,
                 device=torch.device('cpu')):
        super(Actor, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        # GCN layer with 8 input, 16 output
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, feature_dim)
        self.gcn_conv3 = GCNConv(feature_dim, discrete_actions)

        self.max_action = max_action

    # def forward(self, state):
    def forward(self, state, return_mapper=False):

        if isinstance(state.env_features, np.ndarray):
            ev_features = torch.from_numpy(
                state.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(
                state.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(
                state.tr_features).float().to(self.device)
            env_features = torch.from_numpy(
                state.env_features).float().to(self.device)
            edge_index = torch.from_numpy(
                state.edge_index).long().to(self.device)
        else:
            ev_features = state.ev_features
            cs_features = state.cs_features
            tr_features = state.tr_features
            env_features = state.env_features
            edge_index = state.edge_index

        # edge_index = to_undirected(edge_index)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(
            total_nodes, self.feature_dim, device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)
        embedded_x = F.relu(embedded_x)

        # Apply GCN layers with the unified edge index
        x = self.gcn_conv(embedded_x, edge_index)
        x1 = F.relu(x)
        x = self.gcn_conv2(x1, edge_index)
        x = F.relu(x)

        # Residual connection
        # x = embedded_x + x

        x = self.gcn_conv3(x, edge_index)

        # Bound output to action space
        x = self.max_action * torch.tanh(x)
        # x = self.max_action * torch.sigmoid(x)

        # apply action mask
        # valid_action_indexes = torch.where(node_types == 3, 1, 0)
        if self.discrete_actions > 1:
            # print(f'x shape: {x.shape}')
            # print(x)
            x = torch.nn.functional.softmax(x, dim=1)
            # print(f'x shape: {x.shape}')
            # print(x)

        x = x.reshape(-1)
        # input()
        # x = x * valid_action_indexes

        if return_mapper:
            return x, None, state.ev_indexes
        else:
            return x

    def explain_GCN(self, state):

        if isinstance(state.env_features, np.ndarray):
            ev_features = torch.from_numpy(
                state.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(
                state.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(
                state.tr_features).float().to(self.device)
            env_features = torch.from_numpy(
                state.env_features).float().to(self.device)
            edge_index = torch.from_numpy(
                state.edge_index).long().to(self.device)
        else:
            ev_features = state.ev_features
            cs_features = state.cs_features
            tr_features = state.tr_features
            env_features = state.env_features
            edge_index = state.edge_index

        # edge_index = to_undirected(edge_index)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(
            total_nodes, self.feature_dim, device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)
        output_1 = embedded_x.clone().detach().cpu().numpy()

        embedded_x = F.relu(embedded_x)

        # Apply GCN layers with the unified edge index
        x = self.gcn_conv(embedded_x, edge_index)
        output_2 = x.clone().detach().cpu().numpy()

        x1 = F.relu(x)
        # output_3 = x1.clone().detach().cpu().numpy()

        x = self.gcn_conv2(x1, edge_index)
        output_4 = x.clone().detach().cpu().numpy()

        x = F.relu(x)

        x = self.gcn_conv3(x, edge_index)
        output_5 = x.clone().detach().cpu().numpy()

        # Bound output to action space
        x = self.max_action * torch.tanh(x)
        output_6 = F.relu(x).clone().detach().cpu().numpy()

        # apply action mask
        if self.discrete_actions > 1:
            x = torch.nn.functional.softmax(x, dim=1)

        x = x.reshape(-1)

        return [output_1, output_2, output_4, output_6]


class Critic_GNN(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 mlp_hidden_dim=256,
                 discrete_actions=1,
                 device=torch.device('cpu')):
        super(Critic_GNN, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        self.gcn_conv = GCNConv(feature_dim+discrete_actions, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim)
        self.gcn_conv3 = GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim)

        self.l1 = nn.Linear(3*GNN_hidden_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state, action):

        ev_features = state.ev_features
        cs_features = state.cs_features
        tr_features = state.tr_features
        env_features = state.env_features
        edge_index = state.edge_index

        # edge_index = to_undirected(edge_index)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(total_nodes,
                                 self.feature_dim,
                                 device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)
        embedded_x = F.relu(embedded_x)

        # Concatenate action to the embedded_x
        state_action = torch.cat(
            [embedded_x, action.reshape(-1, self.discrete_actions)], 1)

        # Apply GCN layers with the unified edge index
        x = self.gcn_conv(state_action, edge_index)
        x1 = F.relu(x)

        x = self.gcn_conv2(x1, edge_index)
        x = F.relu(x)

        x = self.gcn_conv3(x, edge_index)
        x = F.relu(x)

        # make batch sample mask
        sample_node_length = state.sample_node_length
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        counter = 0
        for i in range(len(sample_node_length)):
            batch[counter: counter + sample_node_length[i]] = i
            counter += sample_node_length[i]

        # Graph Embedding
        pooled_embedding = global_mean_pool(x, batch=batch)

        # Apply MLP
        x = F.relu(self.l1(pooled_embedding))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class Critic(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 mlp_hidden_dim=256,
                 discrete_actions=1,
                 device=torch.device('cpu')):
        super(Critic, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        self.q1 = Critic_GNN(feature_dim=feature_dim,
                             fx_node_sizes=fx_node_sizes,
                             GNN_hidden_dim=GNN_hidden_dim,
                             mlp_hidden_dim=mlp_hidden_dim,
                             discrete_actions=discrete_actions,
                             device=device)

        self.q2 = Critic_GNN(feature_dim=feature_dim,
                             fx_node_sizes=fx_node_sizes,
                             GNN_hidden_dim=GNN_hidden_dim,
                             mlp_hidden_dim=mlp_hidden_dim,
                             discrete_actions=discrete_actions,
                             device=device)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state, action):
        return self.q1(state, action)


class TD3_ActionGNN(object):
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
            lr=3e-4,
            discrete_actions=1,
            **kwargs
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.discrete_actions = discrete_actions

        self.actor = Actor(max_action,
                           feature_dim=fx_dim,
                           GNN_hidden_dim=fx_GNN_hidden_dim,
                           fx_node_sizes=fx_node_sizes,
                           discrete_actions=discrete_actions,
                           device=self.device
                           ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(feature_dim=fx_dim,
                             GNN_hidden_dim=fx_GNN_hidden_dim,
                             mlp_hidden_dim=mlp_hidden_dim,
                             fx_node_sizes=fx_node_sizes,
                             discrete_actions=discrete_actions,
                             device=self.device
                             ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, expl_noise=0, **kwargs):
        state = state.to(self.device)

        with torch.no_grad():
            action, valid_action_indexes, ev_indexes = self.actor(
                state, return_mapper=True)

        noise = torch.randn_like(action) * expl_noise

        action = (action + noise).clamp(-self.max_action, self.max_action)
        # action = action * valid_action_indexes

        mapped_action = np.zeros(self.action_dim)

        if self.discrete_actions == 1:
            for index, i in enumerate(state.action_mapper):
                mapped_action[i] = action[ev_indexes[index]]
        else:
            temp_action = action.reshape(-1, self.discrete_actions)
            # find max index in each row
            temp_action = torch.argmax(temp_action, dim=1)

            for index, i in enumerate(state.action_mapper):
                mapped_action[i] = temp_action[ev_indexes[index]]

        if expl_noise != 0:
            return mapped_action, action
        else:
            return mapped_action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise

            next_action, valid_action_indexes, _ = self.actor_target(next_state,
                                                                     return_mapper=True)

            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip,
                                                                              self.noise_clip)

            next_action = (next_action + noise).clamp(-self.max_action,
                                                      self.max_action)

            # next_action = next_action * valid_action_indexes

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
