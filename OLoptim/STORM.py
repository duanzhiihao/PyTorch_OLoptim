import torch
import time


class STORM(torch.optim.Optimizer):
    '''
    Implements STOchastic Recursive Momentum (STORM)

    Args:
        k (float, optional): learning rate scaling (called k in the original paper).
        c (float, optional): 
        w (float, optional): initial value of denominator in adaptive learning rate
    '''
    def __init__(self, params, c=10, k=0.1, w=0.1):
        defaults = dict(k=k, c=c, w=w)
        super().__init__(params, defaults)
        self.step_num = 0
        self.total_time = 0
    
    def step(self):
        start = time.time()
        inspect = True # inspect the a and eta value of the first weights
        for group in self.param_groups:
            # For different groups, we might want to use different lr, regularizer, ...
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    raise Exception('Please call update_momentum() first. '\
                        'See example_STORM.py for usage.')

                state['step'] += 1
                dt, sum_Gt_sq = state['dt'], state['sum_Gt_sq']
                w, k, c = group['w'], group['k'], group['c']

                eta_t = k / torch.pow(w + sum_Gt_sq, 1/3)
                p.data.add_(-eta_t, dt)
                state['at'] = min(1, c * eta_t**2)
                # if inspect and step_num % 1000 == 0:
                #     print('--mean of eta_t:', eta_t.mean())
                #     print('--a_t:', state['at'])
                inspect = False

        self.step_num += 1
        self.total_time += time.time() - start

    def update_momentum(self):
        for group in self.param_groups:
            for p in group['params']:
                gt = p.grad.data
                if gt is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    assert self.step_num == 0
                    # State initialization
                    state['step'] = 0
                    state['sum_Gt_sq'] = torch.sum(gt*gt)
                    state['recorded_gt'] = torch.zeros_like(p.data)
                    state['dt'] = gt.clone()
                    continue

                gt_prev = state['recorded_gt']
                assert not torch.allclose(gt, gt_prev), 'Please call clone_grad() in ' \
                    'the preious step. See example_STORM.py for usage.'
                state['sum_Gt_sq'] += torch.sum(gt*gt)
                dt = state['dt']
                state['dt'] = gt + (1-state['at'])*(dt - gt_prev)
                # destroy previous cloned gt for safety
                state['recorded_gt'] = None

    def clone_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                gt = p.grad.data
                if gt is None:
                    continue
                self.state[p]['recorded_gt'] = gt.clone().detach()

    def sec_per_step(self):
        '''
        return the average time per update step (seconds)
        '''
        return self.total_time / self.step_num
    
    def get_avg_lr(self):
        # dummy function
        return 0