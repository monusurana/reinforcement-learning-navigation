from agents.dqn_agent import *

import torch
import torch.nn.functional as F

class DoubleDQNAgent(DQNAgent):
    """Derived from DQNAgent and only overriding the learn function."""

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        self.qnetwork_local.eval()
        with torch.no_grad():
            actions_q_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1).long()
            Q_targets_next = self.qnetwork_target(next_states).gather(1, actions_q_local)
        self.qnetwork_local.train()
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)