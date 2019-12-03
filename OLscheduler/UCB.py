import torch
import numpy as np


class UCB(torch.optim.lr_scheduler._LRScheduler):
    """
    Implement the Upper Confidence Bound (UCB) algorithm for automatically tuning \
        learning rate.

    Only support global learning rate.

    Arguments:
        optimizer (iterable): 
    """
    def __init__(self, optimizer, lrs=[0.0001,0.001,0.01,0.1], last_step=-1, alpha=3):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_step = last_step
        
        # self.initial_lr = lr
        self.alpha = alpha
        self._step_count = 1
        self.slots_num = len(lrs)
        self.S = np.zeros(self.slots_num)
        self.cum_loss = np.zeros(self.slots_num)
        self.slots = np.array(lrs)
        self.flag = False

        self._play_one_slot()
    
    def step(self, loss):
        assert loss >= 0
        assert self.flag, 'Locked'
        self.flag = False
        # update S_t
        self.S[self.last_idx] += 1
        # update loss history
        self.cum_loss[self.last_idx] += loss
        # choose a slot machine based on upper confidence bound and play it
        self._play_one_slot()
        self._step_count += 1

    def _play_one_slot(self):
        assert not self.flag, 'Locked'
        self.flag = True
        # calculate upper confidence bound
        St = self.S + 1e-8 if (self.S == 0).any() else self.S
        mu = self.cum_loss / St
        bound = mu - np.sqrt(2*self.alpha*np.log(self._step_count)/St)
        # choose the most confident one
        si = np.argmin(bound)
        self.last_idx = si
        expert_lr = self.slots[si]
        # update learning rate
        self._set_lr(expert_lr)
        # if self._step_count % 1000 == 0:
        #     print(bound)
        #     print(expert_lr)

    def _set_lr(self, lr):
        success = False
        if 'lr' in self.optimizer.state:
            self.optimizer.state['lr'] = lr
            success = True
        for group in self.optimizer.param_groups:
            if 'lr' in group:
                group['lr'] = lr
                success = True
        assert success, 'Failed to set learning rate'