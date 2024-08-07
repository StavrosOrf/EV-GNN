import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete, Discrete

import numpy as np


class BinaryAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] > 0 then action[i] = 1 else action[i] = min_action[i]

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action > 0.5, 1, self.min_action)


class ThreeStep_Action(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == 1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, 0, np.where(action == 1, self.min_action, 1))


class ThreeStep_Action_DiscreteActionSpace(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        num_actions = env.action_space.shape[0]
        env.action_space = MultiDiscrete([3]*env.action_space.shape[0])

        print(f"Action Space: {env.action_space}")

        self.min_action = np.zeros(num_actions)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == 1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, 0, np.where(action == 1, self.min_action, 1))


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Create a mask for the action space to mask the actions that are not available.
    For example, if an EV is not connected to a charging station, then the action to charge the EV is not available.
    """
        
    mask = np.ones((env.action_space.nvec.shape[0],3))    

    counter = 0
    for cs in env.charging_stations:
        for EV in cs.evs_connected:
            if EV is None:
                mask[counter, 1:] = [0] * (env.action_space.nvec[counter] - 1)
            counter += 1
                
    return mask

class Fully_Discrete(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    '''
    This class is used to convert the continuous action space to a discrete action space
    '''
    
    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        num_actions = env.action_space.shape[0]
        print(f"Action Space: {env.action_space}")
              
        new_num_actions = 3**num_actions
        print(f'num_actions: {num_actions}, new_num_actions: {new_num_actions}')  
        env.action_space = Discrete(n=new_num_actions)

        raise NotImplementedError("This class is not implemented yet")
