import time
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import OLoptim
import OLscheduler
from OLutils import utils, models


def example(lr=0.025, algo='SGDOL', bs=4, netname='simple', search=False, sch=None):
    # trainset = utils.MNIST('../MNIST/processed/training.pt')
    # trainset = utils.CIFAR100('./cifar-100-python/train')
    # testset = utils.CIFAR100('./cifar-100-python/test')
    trainset = utils.CIFAR10('./cifar-10-batches-py', trainset=True, enable_aug=True)
    testset = utils.CIFAR10('./cifar-10-batches-py', trainset=False, enable_aug=False)
    dataloader = torch.utils.data.DataLoader(trainset, bs, True, num_workers=0,
                    pin_memory=True)
    dataiter = iter(dataloader)

    if netname == 'simple':
        model = models.SimpleNet(10).cuda() # 'weights/simplenet_init.pth'
    elif netname == 'res18':
        model = models.ResNet(10, layers=[2,2,2,2]).cuda()
    num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('total number of parameters:', num_param)
    # optmz = OLoptim.OSD(model.parameters(), lr=lr) #, V=(-2,2))
    # optmz = OLoptim.FTRL(model.parameters(), V=(-2,2))
    if algo == 'FTML':
        optmz = OLoptim.FTML(model.parameters(), lr=lr)
    elif algo == 'SGDOL':
        optmz = OLoptim.SGDOL(model.parameters())#, momentum=0.9, weight_decay=0.005)
    elif algo == 'SGDPF_global':
        optmz = OLoptim.SGD_globLR(model.parameters())#, momentum=0.9, weight_decay=0.005)
    elif algo == 'SGDPF_cord':
        optmz = OLoptim.SGD_cordLR(model.parameters())#, momentum=0.9, weight_decay=0.005)
    elif algo == 'FTRLP':
        optmz = OLoptim.FTRL_Proximal(model.parameters())#, momentum=0.9, weight_decay=0.005)
    elif algo == 'STORM':
        optmz = OLoptim.STORM(model.parameters(), c=lr)
    elif algo == 'SGD':
        optmz = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    elif algo == 'rawSGD':
        optmz = torch.optim.SGD(model.parameters(), lr=lr)
    elif algo == 'Adam':
        optmz = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception('unknown optimizer')

    # slot_machines = [0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.025,0.1,0.3,1]
    # sche = OLscheduler.EXP3(optmz, lrs=slot_machines)
    # sche = OLscheduler.UCB(optmz, lrs=slot_machines)

    lpath = f'logs/{utils.today()}/{netname}_{algo}{lr:.3g}_{bs}_{utils._now_str()}'
    logger = SummaryWriter(lpath)
    epoch = regret = regret_buff = total_time = correct = incorrect = 0
    T = 100000
    steprange = tqdm(range(T+1)) if search else range(T+1)
    for step in steprange:
        # load \xi_{t+1}
        try:
            imgs, labels = next(dataiter) # load a batch
        except StopIteration:
            epoch += 1
            # print(f'avg optimizer time: {total_time/step:.3g}s')
            dataiter = iter(dataloader)
            imgs, labels = next(dataiter) # load a batch

        imgs, labels = imgs.cuda(), labels.cuda()
        # print('label:', labels[0])
        pred, loss = model(imgs, labels)
        # x = model(imgs)
        # pred = torch.argmax(x,dim=1)
        # loss = torch.nn.functional.cross_entropy(x, labels, reduction='mean')

        correct += (pred == labels).sum().cpu().item()
        incorrect += (pred != labels).sum().cpu().item()

        if algo == 'STORM':
            optmz.zero_grad()
            loss.backward()
            if step == 0:
                optmz.update_momentum()
            else:
                # record the grad for updating momentum
                optmz.clone_grad()
                # put a zero_grad() here just to show that step() only depend on the momentum
                optmz.zero_grad()
                # update to new weights x_{t+1} using previous momemtum
                optmz.step()
                # forward pass using new weights x_{t+1}, calculate \nabla f(x_{t+1}, \xi_{t+1})
                pred, loss = model(imgs, labels)
                # optmz.zero_grad() # the gradients should be already zero
                loss.backward()
                # update the momentum d_t
                optmz.update_momentum()
        else:
            loss.backward()
            tic = time.time()
            optmz.step()
            optmz.zero_grad()
            total_time += time.time() - tic

        loss_ = loss.detach().cpu().item()
        regret += loss_
        regret_buff += loss_

        if step % 500 == 0:
            if not search:
                # learning rate
                if 'lr' in optmz.param_groups[0]:
                    print('Current step size:', optmz.param_groups[0]['lr'])
                elif 'lr' in optmz.state:
                    print('Current step size:', optmz.state['lr'])
                else:
                    print('Current step size:', optmz.get_avg_lr())
            if 'lr' in optmz.param_groups[0]:
                logger.add_scalar('step size', optmz.param_groups[0]['lr'], step)
            elif 'lr' in optmz.state:
                logger.add_scalar('step size', optmz.state['lr'], step)
            else:
                logger.add_scalar('step size', optmz.get_avg_lr(), step)
            # sche.step(regret_buff)
            # logger.add_scalar('schedular_regret', regret_buff, global_step=step)
            regret_buff = 0

        if step > 0 and step % (T//100) == 0:
            logger.add_scalar('loss', loss_, global_step=step)
            logger.add_scalar('regret', regret, global_step=step)
            logger.add_scalar('avg_regret', regret/step, global_step=step)
            # training CCR
            # assert correct + incorrect <= len(trainset)
            ccr = correct/(correct + incorrect)
            logger.add_scalar('Training set accuracy', ccr, step)
            correct = incorrect = 0
            # test CCR
            test_ccr = model.evaluate(testset)
            logger.add_scalar('Test set accuracy', test_ccr, step)
            if not search:
                print(f'[Iteration {step}] [epoch {epoch}] Test set accuracy: {test_ccr}')

    logger.close()
    with open(f'{algo}.txt', 'a') as f:
        print('-----------------------------', utils.now_str(), file=f)
        print(f'model: {netname}, batch size: {bs}, iterations: {step}', file=f)
        print(f'lr={lr}, regret={regret}, avg_regret={regret/step} final test ccr={test_ccr}\n', file=f)
    return regret, test_ccr


if __name__ == "__main__":
    # torch.manual_seed(10)
    regret, ccr = example()
    print('max memory usage:', torch.cuda.max_memory_allocated()/1024/1024/1024, 'GB')


# if __name__ == "__main__":
#     # torch.manual_seed(10)
#     # search for the best fixed learning rate
#     # logger = SummaryWriter(f'logs/{utils.today()}/cifar10_{utils._now_str()}')
#     for p in torch.linspace(-2,2,21):
#         lr = 10**p
#         regret, ccr = example(lr, 'STORM', 4, 'simple', search=True)
#         print(f'lr={lr}, regret={regret}, ccr={ccr}')
#         # logger.add_scalar('Regret', regret, lr)
#         # logger.add_scalar('Test accuracy', ccr, lr)
#     # logger.close()
#     print('max memory usage:', torch.cuda.max_memory_allocated()/1024/1024/1024, 'GB')