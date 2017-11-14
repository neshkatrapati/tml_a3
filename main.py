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
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.autograd import Variable



class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, self.input_size)
    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = x.view(-1, 10)
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


    def step_mdp(self, state, action):
        pstates = self.mdp.possible_states(state,self.mdp.actions[action])
        next_state = list(pstates.keys())[numpy.argmax(numpy.random.multinomial(1, list(pstates.values())))]
        return state, action, next_state, self.mdp.reward(next_state)

    def get_policy(self):
        policy = numpy.zeros((self.mdp.g.size,self.mdp.g.size, self.mdp.action_count))
        for row in range(self.mdp.g.size):
            for col in range(self.mdp.g.size):
                state = (row, col)
                x = g.state_to_mat(state)
                x = Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda())
                output = self.model(x).multinomial()
                action_to_take = output.data.cpu().numpy()[0][0]
                policy[(*state,action_to_take)] = 1.0
        return policy


    def run_once(self, ditch_when = 1000):
        rewards = []
        ditched = False

        current_state = self.mdp.g.start_state
        i = 0
        saved_actions = []
        while (current_state != self.mdp.g.end_state) and i < ditch_when:
            x = g.state_to_mat(current_state)
            x = Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda())
            output = self.model(x).multinomial()
            saved_actions.append(output)

            action_to_take = output.data.cpu().numpy()[0][0]

            state, action, next_state, reward = self.step_mdp(current_state, action_to_take)
            current_state = next_state
            rewards.append(reward)

            i+=1

        rewards = numpy.array(rewards)
        returns = self.rewards_to_returns(rewards)
        mx = numpy.mean(returns)
        stdx = numpy.std(returns)
        if stdx > 0:
            returns = (returns - mx) / stdx # Normalize Returns by the stddev & mean
        returns = numpy.flip(returns, axis=0).copy()

        returns = torch.from_numpy(returns).type(torch.FloatTensor).cuda()

        mx = numpy.mean(rewards)
        stdx = numpy.std(rewards)
        old_rewards = numpy.copy(rewards)
        if stdx > 0:
            rewards = (rewards - mx) / stdx # Normalize Returns by the stddev & mean
        rewards = numpy.flip(rewards, axis=0).copy()
        rewards = torch.from_numpy(rewards).type(torch.FloatTensor).cuda()

        self.optimizer.zero_grad()
        for action, r in zip(saved_actions, returns):
            action.reinforce(r)
            action.backward()

        #autograd.backward(saved_actions, [None for _ in saved_actions])
        self.optimizer.step()
        # if current_state != self.mdp.g.end_state:
        #     #print("Ditching")
        #     return None

        return returns, old_rewards, i




    def run(self, episodes = 10000):
        X, Y, L = [], [], []
        old_policy = None
        policy = []
        #for episode in range(episodes):
        episode = 0
        old_reward = None
        while True:
            episode += 1
            results = self.run_once(ditch_when = 200)
            if results:
                returns, rewards, t = results
                if episode % (1000) == 0:
                    print(episode, returns[-1], numpy.sum(rewards), t)
                    policy = self.get_policy()
                    self.mdp.g.print_grid(policy, print_policy=True)
                    #if numpy.array_equal(old_policy,policy):
                    #    break
                    if old_reward == numpy.sum(rewards):
                        break
                old_reward = numpy.sum(rewards)
                #old_policy = policy




    #     if batch_idx % args.log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader), loss.data[0]))

#train(model)
#qpred = QPredictor(m, soft=True, epsilon=0.05)
#print(qpred.run())
#pi = PolicyIteration(m)
mces1 = MCES(m, soft=True, epsilon=0.2, gamma=0.99)
mces2 = MCES(m, soft=True, epsilon=0.05, gamma=0.99)
sarsa1 = Sarsa(m, soft=True, epsilon = 0.2, gamma=0.99)
sarsa2 = Sarsa(m, soft=True, epsilon = 0.05, gamma=0.99)
q1 = QLearning(m, soft=True, epsilon = 0.2, gamma=0.99)
q2 = QLearning(m, soft=True, epsilon = 0.05, gamma=0.99)
esarsa1 = ExpectedSarsa(m, soft=True, epsilon=0.05, gamma=0.99)
esarsa2 = ExpectedSarsa(m, soft=True, epsilon=0.2, gamma=0.99)
#
# #mces.run()
variants = {mces1:"MC0.2",
            mces2:"MC0.05"}
#variants =   {sarsa1:"Sarsa0.2",
#            sarsa2:"Sarsa0.05"}
#             q1:"Q0.2",
#             q2:"Q0.05",
#             esarsa1:"ESarsa0.05",
#             esarsa2:"ESarsa0.2"}

convergence_points = {}
reward_set = {}
times = 20
for i in range(times):
    for variant, name in variants.items():
        x, y, converged_at = variant.run(debug=False)
        print(variant, "Converged at", converged_at)
        if variant not in convergence_points:
            convergence_points[variant] = 0
            reward_set[variant] = numpy.zeros(10000)
        convergence_points[variant] += converged_at
        reward_set[variant] += y
        if i < times - 1:
            variant.print_info()
        variant.reset()

    print('-'*5)

convergence_points = {variant: c / times for variant, c in convergence_points.items()}
reward_set = {variant: c / times for variant, c in reward_set.items()}

x = range(10000)
for variant, name in variants.items():
    y = reward_set[variant]
    print("Average Convergence of", variant, convergence_points[variant])
    print("Average Reward of", variant, y[-1])
    plt.plot(x, y, label=name)
plt.legend()
plt.show()
