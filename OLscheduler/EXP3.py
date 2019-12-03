import torch
import numpy as np


class EXP3(torch.optim.lr_scheduler._LRScheduler):
    """
    Implement the EXP3 Algorithm for automatically tuning learning rate.

    Only support global learning rate.

    Arguments:
        optimizer (iterable): 
    """
    def __init__(self, optimizer, lrs=[0.0001,0.001,0.01,0.1], last_step=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_step = last_step

        # only support global lr
        # lr = optimizer.state['lr'] if 'lr' in optimizer.state else optimizer.param_groups[0]['lr']
        # for group in optimizer.param_groups:
        #     assert group['lr'] == lr, 'lr in different groups should be the same'
        
        # self.initial_lr = lr
        self._step_count = 1
        self.slots_num = len(lrs)
        self.slots = np.array(lrs)
        self.slot_idx = np.arange(self.slots_num, dtype='int64')
        self.L_inf = 1
        # initialize the probabilities with a uniform distribution
        self.P = np.ones(self.slots_num, dtype='float32') / self.slots_num

        self._play_one_slot()
    
    def step(self, loss):
        assert loss >= 0
        # update L_{\infty}
        self.L_inf = max(self.L_inf, loss)
        # calculate 'step size' \eta
        eta = (np.log(self.slots_num)/self._step_count)**0.5 / self.L_inf
        # estimated loss vector: \hat{g}_t
        gt_estimate = np.zeros(self.slots_num)
        gt_estimate[self.last_idx] = loss / self.P[self.last_idx]
        # update probability distribution
        P = self.P * np.exp(-eta*gt_estimate)
        self.P = P / np.sum(P)

        # draw slot machine At according to distribution P
        self._play_one_slot()
        self._step_count += 1

    def _play_one_slot(self):
        si = np.random.choice(self.slot_idx, size=1, p=self.P).item()
        self.last_idx = si
        expert_lr = self.slots[si]
        self._set_lr(expert_lr)
        if self._step_count % 100 == 0:
            print(self.P)
            print(expert_lr)

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