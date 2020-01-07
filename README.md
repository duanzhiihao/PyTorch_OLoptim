# PyTorch_OLoptim
A Pytorch implementation of various Online / Stochastic optimization algorithms

# Descriptions

**FTRL: Follow the Regularized Leader**
- intro: a classic algorithm in online learning

**FTML: [ICML 2017] Follow the Moving Leader in Deep Learning**
- paper: http://proceedings.mlr.press/v70/zheng17a.html

**SGDOL: [NeurIPS 2019] Surrogate Losses for Online Learning of Stepsizes in Stochastic Non-Convex Optimization**
- intro: automatically tune the learning rate of neural networks by FTRL
- paper: https://arxiv.org/abs/1901.09068
- official implementation: https://github.com/zhenxun-zhuang/SGDOL

**STORM: [NeurIPS 2019] Momentum-Based Variance Reduction in Non-Convex SGD**
- intro: a recently proposed optimizer for neural networks
- paper: https://arxiv.org/abs/1905.10018
- official implementation: https://github.com/google-research/google-research/tree/master/storm_optimizer

**EXP3: Exponential-weight algorithm for Exploration and Exploitation**
- intro: a classic algorithm for (adversarial) multi-armed bandit problem. Implement it as a learning rate scheduler
- original paper: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
- a nice blog post: https://parameterfree.com/2019/11/12/multi-armed-bandit-i/

**UCB: Upper Confidence Bound algorithm**
- intro: a classic algorithm for stochastic multi-armed bandit problem, which achieves the optimal regret bound while is parameter-free. Implement it as a learning rate scheduler
- a nice blog post: https://parameterfree.com/2019/11/21/multi-armed-bandit-iv-ucb/

**SGDPF**
- intro: a toy example to use gradient descent to automatically tune the learning rate. The name comes from 'SGD + parameter free'
