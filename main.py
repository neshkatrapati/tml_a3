import numpy
import random
import sys
import matplotlib.pyplot as plt
from algorithms import *

g = GridWorld(size = 4, start_state=(1,0), bad_state=(2,1), end_state=(3,3))
g.print_grid(None)

m = MDP(g)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size ** 2, 10)
        self.fc2 = nn.Linear(10, self.input_size)
    def forward(self, x):
        x = x.view(-1, self.input_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class QPredictor(MCES):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(QPredictor, self).__init__(mdp, soft, gamma, theta, epsilon)
        self.initialise_model()

    def initialise_model(self):
        self.model = Net(self.mdp.g.size)
        self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)



    # def train(epoch):
    #     model.train()
    #     for batch_idx in range(100):
    #         data = numpy.zeros((32, g.size ** 2))
    #         #target = numpy.zeros((32, g.size))
    #         target = numpy.zeros((32,))
    #         for i in range(32):
    #             rpos = random.randrange(g.size ** 2)
    #             data[i][rpos] = 1.0
    #             #target[i][rpos % g.size] = 1.0
    #             target[i] = rpos % g.size
    #         data, target = torch.from_numpy(data), torch.from_numpy(target)
    #         data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
    #         data, target = data.cuda(), target.cuda()
    #         data, target = Variable(data), Variable(target)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         print(output.data.max(1, keepdim=True)[1], target)
    #
    #         loss = F.nll_loss(output, target)
    #         print(loss)
    #         loss.backward()
    #         optimizer.step()

    def run_once(self, ditch_when = 1000):
        rewards = []
        ditched = False
        X, Y = [], []
        for i, e in enumerate(self.mdp.episode(self.policy)):
            state, action, next_state, reward = e
            X.append(g.state_to_vec(state))
            Y.append(action)
            rewards.append(reward)
            if i > ditch_when:
                print("I am ditching this episode")
                ditched = True
                break

        if ditched:
            return None
        rewards = numpy.array(rewards)
        returns = self.rewards_to_returns(rewards)

        return X, Y, returns

    # def run(self, episodes = 100):
    #     X, Y, L = [], [], []
    #     for episode in range(episodes):
    #         results = self.run_once()
    #         if results:
    #             x, y, l = results
    #             X.extend(x)
    #             Y.extend(y)
    #             L.extend(l)





    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader), loss.data[0]))

#train(model)
qpred = QPredictor(m, soft=True, epsilon=0.05)
qpred.run_once()
#pi = PolicyIteration(m)
# mces1 = MCES(m, soft=True, epsilon=0.2)
# mces2 = MCES(m, soft=True, epsilon=0.05)
# sarsa1 = Sarsa(m, soft=True, epsilon = 0.2)
# sarsa2 = Sarsa(m, soft=True, epsilon = 0.05)
# q1 = QLearning(m, soft=True, epsilon = 0.2)
# q2 = QLearning(m, soft=True, epsilon = 0.05)
# esarsa1 = ExpectedSarsa(m, soft=True, epsilon=0.05)
# esarsa2 = ExpectedSarsa(m, soft=True, epsilon=0.2)
#
# #mces.run()
# variants = {mces1:"MC0.2",
#             mces2:"MC0.05",
#             sarsa1:"Sarsa0.2",
#             sarsa2:"Sarsa0.05",
#             q1:"Q0.2",
#             q2:"Q0.05",
#             esarsa1:"ESarsa0.05",
#             esarsa2:"ESarsa0.2"}
# convergence_points = {}
# reward_set = {}
# times = 20
# for i in range(times):
#     for variant, name in variants.items():
#         x, y, converged_at = variant.run(debug=False)
#         print(variant, "Converged at", converged_at)
#         if variant not in convergence_points:
#             convergence_points[variant] = 0
#             reward_set[variant] = numpy.zeros(10000)
#         convergence_points[variant] += converged_at
#         reward_set[variant] += y
#         variant.reset()
#
#     print('-'*5)
#
# convergence_points = {variant: c / times for variant, c in convergence_points.items()}
# reward_set = {variant: c / times for variant, c in reward_set.items()}
#
# x = range(10000)
# for variant, name in variants.items():
#     y = reward_set[variant]
#     print("Average Convergence of", variant, convergence_points[variant])
#     plt.plot(x, y, label=name)
# plt.legend()
# plt.show()
