# Full credit to https://github.com/zhenxun-zhuang/SGDOL by Zhenxun Zhuang
import torch
import time


class SGDOL_global(torch.optim.Optimizer):
    """Implement the SGDOL Algorithm.
    
    This algorithm was proposed in "Surrogate Losses for Online Learning of \
    Stepsizes in Stochastic Non-Convex Optimization" which can be checked out \
    at: https://arxiv.org/abs/1901.09068

    The official implementation is at https://github.com/zhenxun-zhuang/SGDOL

    The online learning algorithm used here is \
    "Follow-The-Regularized-Leader-Proximal" as described in the paper.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts
            defining parameter groups. Usually = model.parameters()
        smoothness (float, optional): the assumed smoothness of the loss
            function (default: 10).
        alpha (float, optional): the parameter alpha used in the inital
            regularizer, a rule of thumb is to set it as smoothness (default: 10)
    """
    def __init__(self, params, momentum=0, dampening=0, weight_decay=0,
                 smoothness=10.0, alpha=10.0):
        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state['M'] = smoothness
        self.state['cum_grad_normsq'] = alpha
        self.state['cum_inner_prods'] = alpha
        self.state['lr'] = 1.0/smoothness
        self.state['is_first_grad'] = True

        self.step_num = 0
        self.total_time = 0

    def step(self):
        '''Paper version. The notations and formulas follow the convention in the paper.
        '''
        start = time.time()
        lr = self.state['lr']
        if self.state['is_first_grad']:
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    gt = p.grad.data
                    if gt.is_sparse:
                        raise RuntimeError('SGDOL does not support sparse gradients')

                    if weight_decay != 0:
                        gt.add_(weight_decay, p.data)

                    state = self.state[p]
                    state['first_grad'] = gt.clone()

                    if momentum != 0:
                        if 'momentum_buffer' not in state:
                            buf = state['momentum_buffer'] = gt.clone().detach()
                        else:
                            buf = state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, gt)
                        gt = buf
                    p.data.add_(-lr, gt)
        else:
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    gt = p.grad.data
                    if gt.is_sparse:
                        raise RuntimeError('SGDOL does not support sparse gradients')

                    state = self.state[p]
                    gt_prev = state['first_grad']
                    if gt_prev is None:
                        print('Warning: previous gradients are not recorded')
                        continue
                    # Accumulate ||g_t||^2_2
                    self.state['cum_grad_normsq'] += torch.sum(gt_prev*gt_prev).cpu().item()
                    # Accumulate <g_t, g'_t>
                    tmp = torch.dot(gt.flatten(), gt_prev.flatten()).cpu().item()
                    self.state['cum_inner_prods'] += tmp

                    if momentum != 0:
                        assert 'momentum_buffer' in state, 'Please set is_first_grad to True'
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, gt_prev)
                        gt_prev = buf
                    p.data.add_(-lr, gt_prev)

                    state['first_grad'] = None
        
        # Compute the step-size for the next round
        M = self.state['M']
        lr = self.state['cum_inner_prods'] / (M * self.state['cum_grad_normsq'])
        self.state['lr'] = max(min(lr, 2/M), 0)
        self.state['is_first_grad'] = not self.state['is_first_grad']

        self.step_num += 1
        self.total_time += time.time() - start

    def sec_per_step(self):
        '''
        return the average time per update step (seconds)
        '''
        return self.total_time / self.step_num
