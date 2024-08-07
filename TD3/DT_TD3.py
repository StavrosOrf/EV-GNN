
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DT.models.decision_transformer import DecisionTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.sigmoid(self.l3(a))
        return torch.tanh(self.l3(a))


class DT_Actor():

    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 dropout=0.1,
                 embed_dim=128,
                 n_layer=3,
                 n_head=1,
                 max_ep_len=4096,
                 activation_function='relu',
                 K=24,
                 load_path = None,
                 device='cuda'):

        self.max_action = max_action

        self.model = DecisionTransformer(action_tanh=True,
                                         state_dim=state_dim,
                                         act_dim=action_dim,
                                         max_length=K,
                                         max_ep_len=max_ep_len,
                                         hidden_size=embed_dim,
                                         n_layer=n_layer,
                                         n_head=n_head,
                                         n_inner=4*embed_dim,
                                         activation_function=activation_function,
                                         n_positions=1024,
                                         resid_pdrop=dropout,
                                         attn_pdrop=dropout,
                                         ).to(device)

        if load_path is not None:
            model_path = f"{load_path}/model.best"
            self.model.load_state_dict(torch.load(model_path))
        


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DT_Actor_TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device='cuda',
        mode='normal',
        embed_dim=128,
        n_layer=3,
        n_head=1,
        activation_function='relu',
        dropout=0.1,
        K=12,
        max_ep_len=112,
        load_path = None,
        lr = 3e-4
    ):

        self.actor = DT_Actor(state_dim=state_dim,
                              action_dim=action_dim,
                              max_action=max_action,
                              embed_dim=embed_dim,
                              n_layer=n_layer,
                              n_head=n_head,
                              dropout=dropout,
                              activation_function=activation_function,
                              max_ep_len=max_ep_len,
                              K=K,
                              load_path=load_path,
                              device=device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.model.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.model.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = device
        self.K = K
        self.total_it = 0

    def select_action(self, states, actions, rewards, returns_to_go, timesteps):

        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor.model.get_action(states=states,
                                           actions=actions,
                                           rewards=rewards,
                                           returns_to_go=returns_to_go,
                                           timesteps=timesteps)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        self.actor.model.train()
        self.actor_target.model.train()

        s, a, r, d, rtg, timesteps, mask = replay_buffer.get_batch(batch_size=batch_size,
                                                                   max_len=self.K,
                                                                   device=self.device)
        action = a[:, -2, :]
        reward = r[:, -2]
        not_done = (1 - d[:, -2]).reshape(-1, 1)
        state = s[:, -2, :]
        next_state = s[:, -1, :]

        curr_states = s[:, :-1, :]
        curr_actions = a[:, :-1, :]
        curr_rewards = r[:, :-1]
        curr_rewards_to_go = rtg[:, :-1]
        curr_timesteps = timesteps[:, :-1]
        curr_mask = mask[:, :-1]

        next_states = s[:, 1:, :]
        next_actions = a[:, 1:, :]
        next_rewards = r[:, 1:]
        next_rewards_to_go = rtg[:, 1:]
        next_timesteps = timesteps[:, 1:]
        next_mask = mask[:, 1:]

        # Does not work!
        # Adding zero padding to last action and reward
        ##### curr_actions[:, -1, :] = torch.zeros_like(curr_actions[:, -1, :], device=self.device)
        ##### next_actions[:, -1, :] = torch.zeros_like(next_actions[:, -1, :], device=self.device)

        # print(f'curr_states: {curr_states.shape},\n {curr_states}')
        # print(f'next_states: {next_states.shape},\n {next_states}')
        # print(f'curr_actions: {curr_actions.shape},\n {curr_actions}')
        # print(f'action: {action.shape},\n {action}')
        # print(f'next_actions: {next_actions.shape},\n {next_actions}')
        # print(f'curr_rewards: {curr_rewards_to_go.shape},\n {curr_rewards_to_go}')
        # print(f'next_rewards: {next_rewards_to_go.shape},\n {next_rewards_to_go}')
        # print(f'curr_not_dones: {curr_not_dones.shape},\n {curr_not_dones}')
        # print(f'next_not_dones: {next_not_dones.shape},\n {next_not_dones}')
        # print(f'curr_timesteps: {curr_timesteps.shape},\n {curr_timesteps}')
        # print(f'next_timesteps: {next_timesteps.shape},\n {next_timesteps}')
        # print(f'curr_mask: {curr_mask.shape},\n {curr_mask}')
        # print(f'next_mask: {next_mask.shape},\n {next_mask}')

        # print("====================================")
        # print(f'state: {state.shape},\n {state}')
        # print(f'action: {action.shape},\n {action}')
        # print(f'reward: {reward.shape},\n {reward}')
        # print(f'not_done: {not_done.shape},\n {not_done}')
        # print(f'next_state: {next_state.shape},\n {next_state}')
        # print("====================================")

        with torch.no_grad():
            # Select action according to policy and add clipped noise

            _, action_preds, _ = self.actor_target.model.forward(states=next_states,
                                                                 actions=next_actions,
                                                                 rewards=next_rewards,
                                                                 returns_to_go=next_rewards_to_go,
                                                                 timesteps=next_timesteps,
                                                                 attention_mask=next_mask)

            action_preds = action_preds[:, -1]
            noise = (torch.randn_like(action_preds) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                action_preds + noise).clamp(-self.max_action, self.max_action)

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

            _, action_preds, _ = self.actor.model.forward(states=curr_states,
                                                          actions=curr_actions,
                                                          rewards=curr_rewards,
                                                          returns_to_go=curr_rewards_to_go,
                                                          timesteps=curr_timesteps,
                                                          attention_mask=curr_mask)

            # print(f'Predict: action_preds: {action_preds.shape},\n {action_preds[:, -1]}')
            # input()
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, action_preds[:, -1]).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.model.parameters(), self.actor_target.model.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

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
