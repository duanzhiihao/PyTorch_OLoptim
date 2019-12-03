import torch


class OSD(torch.optim.Optimizer):
    '''
    Args:
        params:
        lr (scalar): learning rate. Usually set to the magic number 0.001
        V (tuple, optional): the domain of the network weights.
        L (scalar, optional): the Lipschitz constant of the network weights.
        T (scalar, optional): the number of rounds the algorithm will run.
        optimal_lr (bool, optional): set to True if you want to calculate the optimal
            learning rate by 
    '''
    def __init__(self, params, lr, V=(None,None), L=None, T=None, optimal_lr=True):
        super().__init__(params, dict())
        pmin, pmax = V
        num_params = 0
        for group in self.param_groups:
            for p in group['params']:
                num_params += p.numel()
        if pmin and pmax:
            assert pmax > pmin, 'the domain is not feasible'
            L_inf = pmax - pmin
            D = (num_params*(L_inf**2))**0.5
            print('Diameter of the domain is', D)
        else:
            D = None
        if D and L and T and optimal_lr:
            lr = D/(L * T**0.5)
            print('Using optimal step size:', lr)
        self.pmin, self.pmax = pmin, pmax # bounded domain
        self.state['d'] = num_params # dimensions
        self.state['D'] = D # Diameter w.r.t. 2-norm
        self.state['lr'] = lr # step size
        self.state['step'] = 0 # number of iterations
        self.state['gt2_sum'] = 0 # \sum_{t=1}^T 0.5 \eta \|g_t\|_2^2

    def step(self):
        self.cur_p = self.cur_g = None
        lr = self.state['lr']
        gt_all = []
        # p_all = []
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p in group['params']:
                gt_p = p.grad
                if gt_p is None:
                    print('skip one layer')
                    continue
                gt_all.append(gt_p.flatten())
                p.data.add_(- lr*gt_p)
                if self.pmin or self.pmax: # projection
                    p.data.clamp_(min=self.pmin, max=self.pmax)
        # logging
        gt_all = torch.cat(gt_all, dim=0)
        gt_L2 = gt_all.pow(2).sum().cpu().item()
        self.state['cur_gt2norm'] = gt_L2**0.5
        self.state['gt2_sum'] += 0.5*lr*gt_L2
        self.state['step'] += 1
        # save parameters and gradients if needed
        # self.cur_p = torch.cat(p_all) if hp else None
        # self.cur_g = torch.histc(gt_all, bins=100) if hg else None
