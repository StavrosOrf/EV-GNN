import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from TD3.TD3_GNN import GNN_Feature_Extractor
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import global_mean_pool
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork_GNN_FX(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim,
                 fx_node_sizes,
                 fx_dim=8,
                 fx_GNN_hidden_dim=32,
                 num_heads=2,):

        super(QNetwork_GNN_FX, self).__init__()

        self.feature_extractor = GNN_Feature_Extractor(feature_dim=fx_dim,
                                                       fx_node_sizes=fx_node_sizes,
                                                       GNN_hidden_dim=fx_GNN_hidden_dim,
                                                       num_heads=num_heads,
                                                       device=torch.device('cuda'))

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = self.feature_extractor(state)

        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy_GNN_FX(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim,
                 action_space,
                 fx_node_sizes,
                 fx_dim=8,
                 fx_GNN_hidden_dim=32,
                 num_heads=2,):

        super(GaussianPolicy_GNN_FX, self).__init__()

        self.feature_extractor = GNN_Feature_Extractor(feature_dim=fx_dim,
                                                       fx_node_sizes=fx_node_sizes,
                                                       GNN_hidden_dim=fx_GNN_hidden_dim,
                                                       num_heads=num_heads,
                                                       device=torch.device('cuda'))

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):

        state = self.feature_extractor(state)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_GNN_FX, self).to(device)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim,
                 action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class GaussianPolicy_ActionGNN(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 action_space=None,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 device=torch.device('cpu')):
        super(GaussianPolicy_ActionGNN, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        # GCN layer with 8 input, 16 output
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, feature_dim)
        self.mean_linear = GCNConv(feature_dim, 1)

        self.log_std_linear = GCNConv(feature_dim, 1)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

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

        mean = self.mean_linear(x, edge_index)
        log_std = self.log_std_linear(x, edge_index)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        # apply action mask
        # valid_action_indexes = torch.where(node_types == 3, 1, 0)
        x = x.reshape(-1)
        # x = x * valid_action_indexes

        if return_mapper:
            return mean, log_std, state.ev_indexes
        else:
            return mean, log_std, None

    def sample(self, state, return_mapper=False):

        mean, log_std, mapper = self.forward(state, return_mapper)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        action_scale = self.action_scale[0]*torch.ones_like(y_t)
        action_bias = self.action_bias[0]*torch.ones_like(y_t)

        action = y_t * action_scale + action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)

        # make batch sample mask
        sample_node_length = state.sample_node_length
        batch = torch.zeros(
            log_prob.shape[0], dtype=torch.long, device=self.device)

        counter = 0
        for i in range(len(sample_node_length)):
            batch[counter: counter + sample_node_length[i]] = i
            counter += sample_node_length[i]

        log_prob = global_add_pool(log_prob, batch=batch)
        mean = torch.tanh(mean) * action_scale + action_bias

        if return_mapper:
            return action, log_prob, mean, mapper
        else:
            return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_ActionGNN, self).to(device)


class Critic_GNN(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 mlp_hidden_dim=256,
                 device=torch.device('cpu')):
        super(Critic_GNN, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        self.q1 = Critic_ActionGNN(feature_dim=feature_dim,
                                   fx_node_sizes=fx_node_sizes,
                                   GNN_hidden_dim=GNN_hidden_dim,
                                   mlp_hidden_dim=mlp_hidden_dim,
                                   device=device)

        self.q2 = Critic_ActionGNN(feature_dim=feature_dim,
                                   fx_node_sizes=fx_node_sizes,
                                   GNN_hidden_dim=GNN_hidden_dim,
                                   mlp_hidden_dim=mlp_hidden_dim,
                                   device=device)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


class Critic_ActionGNN(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 mlp_hidden_dim=256,
                 device=torch.device('cpu')):
        super(Critic_ActionGNN, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN and GAT layers to extract features with a unified edge index
        # GCN layer with 8 input, 16 output
        self.gcn_conv = GCNConv(feature_dim+1, GNN_hidden_dim)
        self.gcn_conv2 = GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim)
        self.gcn_conv3 = GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim)

        self.l1 = nn.Linear(3*GNN_hidden_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

        self.apply(weights_init_)

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
            [embedded_x, action.reshape(-1, 1)], 1)

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