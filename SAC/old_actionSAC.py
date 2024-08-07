import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from SAC.utils import soft_update, hard_update
from SAC.old_model import GaussianPolicy_ActionGNN, Critic_GNN


class SAC_ActionGNN(object):

    algo_name = "SAC ActionGNN"

    def __init__(self,
                 action_space,
                 fx_node_sizes,
                 args):

        self.gamma = args['discount']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']

        self.device = args['device']
        self.hidden_size = args['hidden_size']
        self.lr = args['lr']

        self.action_dim = action_space.shape[0]

        # Using GNN feauture extractor

        self.critic = Critic_GNN(feature_dim=args['fx_dim'],
                                 fx_node_sizes=fx_node_sizes,
                                 GNN_hidden_dim=args['fx_GNN_hidden_dim'],
                                 mlp_hidden_dim=self.hidden_size,
                                 device=self.device).to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = Critic_GNN(feature_dim=args['fx_dim'],
                                        fx_node_sizes=fx_node_sizes,
                                        GNN_hidden_dim=args['fx_GNN_hidden_dim'],
                                        mlp_hidden_dim=self.hidden_size,
                                        device=self.device).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - \
                    torch.prod(torch.Tensor(
                        action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy_ActionGNN(action_space=action_space,
                                                   fx_node_sizes=fx_node_sizes,
                                                   feature_dim=args['fx_dim'],
                                                   GNN_hidden_dim=args['fx_GNN_hidden_dim'],
                                                   device=self.device).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        else:
            raise ValueError("Only Gaussian policy is supported")

    def select_action(self, state, evaluate=True, return_mapped_action=False):

        if evaluate is False:

            if return_mapped_action:
                action, _, _, ev_indexes = self.policy.sample(state,
                                                              return_mapper=True)
                mapped_action = np.zeros(self.action_dim)

                for index, i in enumerate(state.action_mapper):
                    mapped_action[i] = action[ev_indexes[index]]

                return mapped_action, action.detach()
            else:
                raise ValueError("Only evaluate mode is supported for now")
                action, _, _ = self.policy.sample(state)
                return action.detach().cpu().numpy()[0]
        else:
            with torch.no_grad():
                if return_mapped_action:
                    _, _, action, ev_indexes = self.policy.sample(state,
                                                                  return_mapper=True)
                    mapped_action = np.zeros(self.action_dim)

                    for index, i in enumerate(state.action_mapper):
                        mapped_action[i] = action[ev_indexes[index]]

                    return mapped_action
                else:
                    raise ValueError("Only evaluate mode is supported for now")
                    _, _, action = self.policy.sample(state)

        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, not_dones = memory.sample(
            batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + not_dones * \
                self.gamma * (min_qf_next_target)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
                           self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save(self, save_path):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models to {}'.format(save_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, save_path)

    # Load model parameters
    def load(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(
                checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()