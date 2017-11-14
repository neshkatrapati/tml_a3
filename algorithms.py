import numpy
import random


class GridWorld(object):
    def __init__(self, size, start_state, bad_state, end_state):
        self.size = size
        self.start_state = start_state
        self.bad_state = bad_state
        self.end_state = end_state
        self.grid = numpy.zeros((size, size))

    def print_grid(self, policy, print_policy = False):
        labels = ['←', '→', '↑', '↓']
        for row in range(self.size):
            for col in range(self.size):
                state_symbol = ''
                if (row, col) == self.start_state:
                    state_symbol = 'S'
                elif (row, col) == self.bad_state:
                    state_symbol = 'B'
                elif (row, col) == self.end_state:
                    state_symbol = 'G'
                elif not print_policy:
                    state_symbol = '-'
                if print_policy:
                    state_symbol += labels[numpy.argmax(policy[(row,col)])]
                print(state_symbol, end=' ')
            print('')

    def state_to_vec(self, state):
        index = state[0] * self.size + state[1]
        v = numpy.zeros(self.size**2)
        v[index] = 1
        return v



class MDP(object):
    def __init__(self, g, a = 0.9, b =0.05):
        self.g = g
        self.action_count = 4 # Left,Right,Up,Down
        self.actions = ['left', 'right', 'up', 'down']

        self.a, self.b = a, b

        self.lateral = {'up' : ['left', 'right'], 'left': ['up', 'down']}
        self.lateral['right'] = self.lateral['left']
        self.lateral['down'] = self.lateral['up']



    def episode(self, policy):
        current_state = self.g.start_state
        not_done = True
        while not_done:
            if current_state == self.g.end_state:
                not_done = False
            action_to_take = numpy.argmax(numpy.random.multinomial(1,policy[current_state]))
            pstates = self.possible_states(current_state,self.actions[action_to_take])
            next_state = list(pstates.keys())[numpy.argmax(numpy.random.multinomial(1, list(pstates.values())))]
            yield current_state, action_to_take, next_state, self.reward(next_state)
            current_state = next_state

        #yield current_state, action_to_take, next_state, self.reward(next_state)


    def possible_states(self, state, action):
        states = {}

        states['left'] = state[0], max(state[1] -1, 0)
        states['right'] = state[0], min(state[1] + 1, self.g.size - 1)
        states['up'] = max(state[0] - 1,0), state[1]
        states['down'] = min(state[0] + 1, self.g.size - 1), state[1]
        if state == self.g.end_state:
            return {state:1} # End State is Absorbing

        pstates={states[action]:self.a}
        for l in self.lateral[action]:
            if states[l] not in pstates:
                pstates[states[l]] = 0
            pstates[states[l]] += self.b

        return pstates

    def reward(self, state):
        if state == self.g.end_state:
            return 100
        elif state == self.g.bad_state:
            return -70
        elif state == self.g.start_state:
            return 0
        else:
            return -1



class PolicyIteration(object):

    def __init__(self, mdp, gamma = 0.9, theta=0.05):
        self.mdp = mdp
        self.state_values = numpy.zeros((self.mdp.g.size, self.mdp.g.size))
        self.policy = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        self.gamma = gamma
        self.theta = theta
        self.initialise_deterministic_policy()
        print(self.state_values)
        print(self.policy)


    def initialise_deterministic_policy(self):
        for row in range(self.mdp.g.size):
            for col in range(self.mdp.g.size):
                self.policy[row,col,random.randrange(self.mdp.action_count)] = 1.0


    def state_action_value(self, state, action):
        next_states = self.mdp.possible_states(state, action)
        estimate = 0
        for next_state, prob in next_states.items():
            reward = self.mdp.reward(next_state)
            old_state_value = self.state_values[next_state] # vpi(s')
            estimate += (reward + self.gamma * old_state_value) * prob
        return estimate


    def state_value(self, state):

        transition_prob = 0
        for i, action in enumerate(self.mdp.actions):
            estimate = self.state_action_value(state, action)
            transition_prob += self.policy[(*state, i)] * estimate
        return transition_prob



    def policy_evaluation(self):
        delta = 1000
        i = 1
        while delta > self.theta:
            for row in range(self.mdp.g.size):
                for col in range(self.mdp.g.size):
                    state = (row, col)
                    old_value = self.state_values[state]
                    self.state_values[state] = self.state_value(state)
                    delta = min(abs(self.state_values[state] - old_value), delta)
            i+=1


    def policy_improvement(self):
        policy_stable = False
        itr = 0
        while policy_stable == False:
            policy_stable = False
            for row in range(self.mdp.g.size):
                for col in range(self.mdp.g.size):
                    state = (row, col)
                    old_action = numpy.argmax(self.policy[state])
                    estimates = numpy.zeros(self.mdp.action_count)
                    for i, action in enumerate(self.mdp.actions):
                        estimates[i] = self.state_action_value(state, action)

                    new_action = numpy.argmax(estimates)

                    if new_action == old_action:

                        policy_stable = True
                    else:
                        policy_stable = False
                        self.policy[state] = self.policy[state] * 0
                        self.policy[(*state,new_action)] = 1 # Update Greedy
                        self.policy_evaluation()
                        #transition_prob = policy[(*next_state, i)] * estimate
                itr += 1
        print("Completed Policy Iteration in ", itr, "Iterations")



class SamplingAlgorithm(object):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        self.mdp = mdp
        self.state_action_values = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        self.state_action_counts = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        self.gamma, self.theta, self.soft, self.epsilon = gamma, theta,soft, epsilon

        self.initialise_deterministic_policy()

        print(self.state_action_values)
        print(self.policy)


    def initialise_deterministic_policy(self):
        self.policy = numpy.ones((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count)) * (self.epsilon / self.mdp.action_count)
        for row in range(self.mdp.g.size):
            for col in range(self.mdp.g.size):
                self.policy[row,col,random.randrange(self.mdp.action_count)] += 1.0 - self.epsilon
        self.policy[self.mdp.g.end_state] =  self.policy[self.mdp.g.end_state] * 0.0

    def update_policy(self):
        if not self.soft:
            self.policy = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        else:
            self.policy = numpy.ones((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count)) * (self.epsilon / self.mdp.action_count)
        for row in range(self.state_action_values.shape[0]):
            for col in range(self.state_action_values.shape[1]):
                action = numpy.argmax(self.state_action_values[row,col])
                if self.soft:
                    self.policy[row,col,action] += 1 - self.epsilon
                else:
                    self.policy[row,col,action] = 1


    def reset(self):
        self.state_action_values = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        self.state_action_counts = numpy.zeros((self.mdp.g.size, self.mdp.g.size, self.mdp.action_count))
        self.initialise_deterministic_policy()

    def run(self, debug = True):
        limit = 10000
        y = []
        prev_pol = None
        converged = False
        converged_at = None
        for i in range(limit):
            rewards = self.run_once()
            if i % 1000 == 0 and debug and rewards:
                self.mdp.g.print_grid(self.policy, print_policy=True)
                print(sum(rewards))
                sys.stdout.write('\r----\n')
            if not numpy.array_equal(prev_pol, self.policy):
                prev_pol = numpy.copy(self.policy)
            elif not converged:
                if debug:
                    print(self.__str__(), "Converged at iteration", i)
                converged = True
                converged_at = i

            y.append(sum(rewards))

        return range(limit), y, converged_at


    def __str__(self):
        return self.__class__.__name__ + " epsilon={epsilon}".format(epsilon = self.epsilon)



class MCES(SamplingAlgorithm):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(MCES, self).__init__(mdp, soft, gamma, theta, epsilon)

    def rewards_to_returns(self, rewards):
        G = numpy.array([self.gamma**i for i in range(len(rewards))])
        rlen = len(rewards)
        returns = []
        for i, r in enumerate(rewards):
            rg = rewards[i:] * G[:rlen-i]
            cs = numpy.cumsum(rg)
            returns.append(cs[-1])
        return returns

    def update_state_action_values(self, state_firsts, returns):
        for sa_pair, first in state_firsts.items():
            rtrn = returns[first]
            self.state_action_values[sa_pair] = ((self.state_action_values[sa_pair] * self.state_action_counts[sa_pair]) + rtrn)/(self.state_action_counts[sa_pair] + 1)
            self.state_action_counts[sa_pair] += 1


    def run_once(self, ditch_when = 1000):
        rewards = []
        state_firsts = {}
        ditched = False
        for i, e in enumerate(self.mdp.episode(self.policy)):
            state, action, next_state, reward = e
            if (*state, action) not in state_firsts:
                state_firsts[(*state, action)] = i # Record the first appearance of the (s,a) pair
            rewards.append(reward)
            if i > ditch_when:
                print("I am ditching this episode")
                ditched = True
                break


        rewards = numpy.array(rewards)
        returns = self.rewards_to_returns(rewards)
        self.update_state_action_values(state_firsts, returns)

        self.update_policy()

        return rewards


class TDAlgorithm(SamplingAlgorithm):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(TDAlgorithm, self).__init__(mdp, soft, gamma, theta, epsilon)


    def td_estimate(self, state):
        return None

    def run_once(self):
        rewards = []

        for i, e in enumerate(self.mdp.episode(self.policy)):
            state, action, next_state, reward = e
            td_estimate = self.td_estimate(next_state)
            alpha = ( self.state_action_counts[(*state, action)] + 1 )** -1
            uvalue = (1-alpha)*self.state_action_values[(*state, action)] + alpha * (reward + self.gamma * td_estimate)
            self.state_action_values[(*state, action)] = uvalue
            self.state_action_counts[(*state, action)] += 1

            rewards.append(reward)

        rewards = numpy.array(rewards)

        self.update_policy()
        return rewards




class Sarsa(TDAlgorithm):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(Sarsa, self).__init__(mdp, soft, gamma, theta, epsilon)

    def td_estimate(self, state):
        next_action = numpy.argmax(numpy.random.multinomial(1, self.policy[state]))
        return self.state_action_values[(*state, next_action)]


class ExpectedSarsa(TDAlgorithm):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(ExpectedSarsa, self).__init__(mdp, soft, gamma, theta, epsilon)

    def td_estimate(self, state):
         expected_value = numpy.sum(self.policy[state] * self.state_action_values[state])
         return expected_value



class QLearning(TDAlgorithm):
    def __init__(self, mdp, soft=False, gamma = 0.9, theta=0.05, epsilon = 0.05):
        super(QLearning, self).__init__(mdp, soft, gamma, theta, epsilon)

    def td_estimate(self, state):
        next_action = numpy.argmax(self.policy[state])
        return self.state_action_values[(*state, next_action)]
