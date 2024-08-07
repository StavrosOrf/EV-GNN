import numpy as np
import torch
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class GNN_ReplayBuffer(object):
    def __init__(self, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # state is a dict of type Data
        self.state = [{} for i in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.next_state = [{} for i in range(max_size)]
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0
        ev_features = np.concatenate(
            [self.state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.state[i].node_types) for i in ind]

        for i in ind:
            edge_index.append(self.state[i].edge_index + edge_counter)
            ev_indexes = np.concatenate(
                [ev_indexes, self.state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.state[i].node_types)
            if self.state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)
        state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                           ev_features=torch.from_numpy(
                               ev_features).to(self.device).float(),
                           cs_features=torch.from_numpy(
                               cs_features).to(self.device).float(),
                           tr_features=torch.from_numpy(
                               tr_features).to(self.device).float(),
                           env_features=torch.from_numpy(
                               env_features).to(self.device).float(),
                           node_types=torch.from_numpy(
                               node_types).to(self.device).float(),
                           sample_node_length=sample_node_length,
                           ev_indexes=ev_indexes,
                           cs_indexes=cs_indexes,
                           tr_indexes=tr_indexes,
                           env_indexes=env_indexes)

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0
        ev_features = np.concatenate(
            [self.next_state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.next_state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.next_state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.next_state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.next_state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.next_state[i].node_types) for i in ind]
        for i in ind:
            edge_index.append(self.next_state[i].edge_index + edge_counter)

            ev_indexes = np.concatenate(
                [ev_indexes, self.next_state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.next_state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.next_state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.next_state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.next_state[i].node_types)
            if self.next_state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.next_state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)
        next_state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                                ev_features=torch.from_numpy(
                                    ev_features).to(self.device).float(),
                                cs_features=torch.from_numpy(
                                    cs_features).to(self.device).float(),
                                tr_features=torch.from_numpy(
                                    tr_features).to(self.device).float(),
                                env_features=torch.from_numpy(
                                    env_features).to(self.device).float(),
                                node_types=torch.from_numpy(
                                    node_types).to(self.device),
                                sample_node_length=sample_node_length,
                                ev_indexes=ev_indexes,
                                cs_indexes=cs_indexes,
                                tr_indexes=tr_indexes,
                                env_indexes=env_indexes)
        return (
            state_batch,
            torch.FloatTensor(self.action[ind]).to(self.device),
            next_state_batch,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class ActionGNN_ReplayBuffer(object):
    def __init__(self, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # state is a dict of type Data
        self.state = [{} for i in range(max_size)]
        self.action = [{} for i in range(max_size)]
        self.next_state = [{} for i in range(max_size)]
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0

        ev_features = np.concatenate(
            [self.state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.state[i].node_types) for i in ind]

        for i in ind:
            edge_index.append(self.state[i].edge_index + edge_counter)
            ev_indexes = np.concatenate(
                [ev_indexes, self.state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.state[i].node_types)
            if self.state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)

        state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                           ev_features=torch.from_numpy(
                               ev_features).to(self.device).float(),
                           cs_features=torch.from_numpy(
                               cs_features).to(self.device).float(),
                           tr_features=torch.from_numpy(
                               tr_features).to(self.device).float(),
                           env_features=torch.from_numpy(
                               env_features).to(self.device).float(),
                           node_types=torch.from_numpy(
                               node_types).to(self.device).float(),
                           sample_node_length=sample_node_length,
                           ev_indexes=ev_indexes,
                           cs_indexes=cs_indexes,
                           tr_indexes=tr_indexes,
                           env_indexes=env_indexes)

        action_batch = torch.concatenate([self.action[i] for i in ind], axis=0)

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0
        ev_features = np.concatenate(
            [self.next_state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.next_state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.next_state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.next_state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.next_state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.next_state[i].node_types) for i in ind]

        for i in ind:
            edge_index.append(self.next_state[i].edge_index + edge_counter)
            ev_indexes = np.concatenate(
                [ev_indexes, self.next_state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.next_state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.next_state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.next_state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.next_state[i].node_types)
            if self.next_state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.next_state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)
        next_state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                                ev_features=torch.from_numpy(
                                    ev_features).to(self.device).float(),
                                cs_features=torch.from_numpy(
                                    cs_features).to(self.device).float(),
                                tr_features=torch.from_numpy(
                                    tr_features).to(self.device).float(),
                                env_features=torch.from_numpy(
                                    env_features).to(self.device).float(),
                                node_types=torch.from_numpy(
                                    node_types).to(self.device),
                                sample_node_length=sample_node_length,
                                ev_indexes=ev_indexes,
                                cs_indexes=cs_indexes,
                                tr_indexes=tr_indexes,
                                env_indexes=env_indexes)

        return (
            state_batch,
            action_batch,
            next_state_batch,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class DT_ReplayBuffer(object):
    def __init__(self, state_dim,
                 action_dim,
                 max_episode_length,
                 max_size=int(1e6),
                 state_mean=None,
                 state_std=None,
                 scale=1.0):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_length = max_episode_length

        # covert to numpy
        self.state_mean = state_mean.detach().cpu().numpy()
        self.state_std = state_std.detach().cpu().numpy()

        self.scale = scale

        self.states = np.zeros((max_size, max_episode_length, state_dim))
        self.actions = np.zeros((max_size, max_episode_length, action_dim))
        self.rewards = np.zeros((max_size, max_episode_length))
        self.not_dones = np.zeros((max_size, max_episode_length))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, states, actions, rewards, dones):
        # TODO: this doesn't work for episodes that finish earlier or later than max_episode_length
        self.states[self.ptr, :, :] = states.detach().cpu().numpy()
        self.actions[self.ptr, :, :] = actions.detach().cpu().numpy()
        self.rewards[self.ptr, :] = rewards.detach().cpu().numpy()
        self.not_dones[self.ptr, :] = 1. - dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self, batch_size=256, max_len=12, device='cuda'):

        batch_inds = np.random.randint(0, self.size, size=batch_size)
        max_len = max_len + 1  # for the state + next state modelling

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for index in batch_inds:

            si = random.randint(0, np.where(self.not_dones[index] == 0)[0] - 1)

            # get sequences from dataset
            s.append(self.states[index, si:si + max_len,
                     :].reshape(1, -1, self.state_dim))
            a.append(self.actions[index, si:si + max_len,
                                  ].reshape(1, -1, self.action_dim))
            r.append(self.rewards[index, si:si + max_len].reshape(1, -1, 1))

            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            d.append(1 - self.not_dones[index, si:si + max_len].reshape(1, -1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >=
                          self.max_episode_length] = self.max_episode_length-1  # padding cutoff
            rtg.append(discount_cumsum(self.rewards[index, si:], gamma=1.)[
                       :s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1],
                                         np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, max_len -
                                   tlen, self.action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen))
                                   * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),
                                     rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        # print(f's: {s}')
        # print(f'a: {a}')
        # print(f'r: {r[0,:]}')
        # print(f'd: {d}')
        # print(f'rtg: {rtg[0,:]}')
        # print(f'timesteps: {timesteps}')
        # print(f'mask: {mask}')
        # input()
        # print(f's.shape: {s.shape}')
        # print(f'a.shape: {a.shape}')
        # print(f'r.shape: {r.shape}')
        # print(f'd.shape: {d.shape}')
        # print(f'rtg.shape: {rtg[:,:-1].shape}')
        # print(f'timesteps.shape: {timesteps.shape}')
        # print(f'mask.shape: {mask.shape}')
        return s, a, r, d, rtg[:, :-1], timesteps, mask
