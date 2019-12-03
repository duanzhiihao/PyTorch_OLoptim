import torch
import time


class FTRL_Proximal(torch.optim.Optimizer):
    '''
    Follow The Learder with Quadratic losses a.k.a. Follow The Regularized Leader \
        Proximal (FTRL-Proximal) without the regularizer

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        
    '''
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-10, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.step_num = 0
        self.total_time = 0

    def step(self):
        '''Paper version. The notations and formulas follow the convention in the paper.
        '''
        start = time.time()
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            beta1, beta2 = group['betas']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                gt = p.grad.data

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['mt'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['vt'] = torch.zeros_like(p.data) + group['eps']
                    state['sum_vt'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                mt, vt, sum_vt = state['mt'], state['vt'], state['sum_vt']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    gt.add_(group['weight_decay'], p.data)
                
                mt.mul_(beta1).add_(1-beta1, gt)
                vt.mul_(beta2).addcmul_(1-beta2, gt, gt)
                sum_vt.add_(torch.sqrt(vt/bias_correction2))

                p.data.add_(-lr*gt / sum_vt / bias_correction1)
                
        self.step_num += 1
        self.total_time += time.time() - start

    def sec_per_step(self):
        '''
        return the average time per update step (seconds)
        '''
        return self.total_time / self.step_num


class FTRL_Linear(torch.optim.Optimizer):
    '''
    Follow The Regularized Learder with linearized losses

    Warning: totally doesn't work

    Args:
        params:
        lr (scalar): learning rate. Usually set to the magic number 0.001
        V (tuple, optional): the domain of the network weights.
        L (scalar, optional): the Lipschitz constant of the network weights.
    '''
    def __init__(self, params, V=(None,None), L=1):
        defaults = dict()
        super().__init__(params, defaults)
        pmin, pmax = V
        num_params = 0
        for group in self.param_groups:
            for p in group['params']:
                num_params += p.numel()
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.zeros_like(p.data)
        if pmin and pmax:
            assert pmax > pmin, 'the domain is not feasible'
            L_inf = pmax - pmin
            D = (num_params*(L_inf**2))**0.5
            print('Diameter of the domain is', D)
        else:
            D = None
        self.pmin, self.pmax = pmin, pmax # bounded domain
        # self.state['d'] = num_params # dimensions
        # self.state['D'] = D # Diameter w.r.t. 2-norm
        self.state['L'] = L # Lipschitz constant

    def step(self):
        '''
        Args:
            func: nabla psi^* at this iteration, i.e., the derivative of the Fenchel
                conjugate of the regularizer. It's better to be a strongly-convex function.
                (default: derivative of 1/2 squared 2-norm)
        '''
        gt_all = []
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p in group['params']:
                gt_p = p.grad
                if gt_p is None:
                    print('skip one layer')
                    continue
                state = self.state[p]
                state['step'] += 1
                state['sum'].add_(-gt_p)

                L = self.state['L']
                t = state['step'] 
                p.data = FTRL_Linear.mirror_func(state['sum'], L, t)

                gt_all.append(gt_p.flatten())
                if self.pmin or self.pmax: # projection
                    p.data.clamp_(min=self.pmin, max=self.pmax)
        # logging
        gt_all = torch.cat(gt_all, dim=0)
        L = gt_all.pow(2).sum().cpu().item()**0.5
        self.state['L'] = max(self.state['L'], L)

    @staticmethod
    def mirror_func(theta_t, L, t):
        '''
        this is the update rule correspond to psi(x) = L*t**0.5/(mu**0.5) * 0.5*||x||_2^2
        '''
        xt = theta_t / (L * t**0.5)
        return xt
