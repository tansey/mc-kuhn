import random
import numpy as np
import matplotlib.pyplot as plt

CHECK_OR_FOLD = 0
BET_OR_CALL = 1

NAMES = ['Jack', 'Queen', 'King']
RUN_ID = 'self-play'

def play_kuhn_hand(agent1, agent2):
    cards = random.sample([0,1,2],2)
    card1 = cards[0]
    card2 = cards[1]
    action1 = agent1.action(card1)
    if not action1:
        if card1 > card2: # check, P1 wins
            return (1,-1, False)
        return (-1, 1, False) # check, P2 wins
    else:
        action2 = agent2.action(card2)
        if not action2: # fold
            return (1, -1, True)
        if card1 > card2: # call, P1 wins
            return (2,-2, True)
        return (-2, 2, True) # call, P2 wins

class Agent(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.valuefn = [[[i / 100.0, 0, 0] for i in range(101)] for _ in range(3)]
        self.exploit = False

    def action(self, card):
        if not self.exploit and random.random() < self.epsilon:
            self.prev = (card, self.random(card), self.epsilon / len(self.valuefn[card]))
        else:
            self.prev = (card, self.greedy(card), (1.0 - self.epsilon) / len(self.valuefn[card]))
        prob = self.valuefn[card][self.prev[1]][0]
        if random.random() < prob:
            return BET_OR_CALL
        return CHECK_OR_FOLD

    def reward(self, r):
        if self.exploit:
            return
        vprev = self.valuefn[self.prev[0]][self.prev[1]][1]
        self.valuefn[self.prev[0]][self.prev[1]][2] += self.prev[2]
        self.valuefn[self.prev[0]][self.prev[1]][1] += (r -  vprev) * self.prev[2] / self.valuefn[self.prev[0]][self.prev[1]][2]
        #self.valuefn[self.prev[0]][self.prev[1]][1] += (r -  vprev) * 0.1

    def greedy(self, card):
        maxi = 0
        maxv = self.valuefn[card][maxi]
        for i in range(len(self.valuefn[card])):
            if self.valuefn[card][i][1] > maxv[1]:
                maxi = i
                maxv = self.valuefn[card][i]
        return maxi

    def greedyprob(self, card):
        i = self.greedy(card)
        return self.valuefn[card][i][0]

    def random(self, card):
        available = self.valuefn[card]
        return random.randint(0,len(available)-1)

class LearningCurve(object):
    def __init__(self, agent, action_name):
        self.agent = agent
        self.x = []
        self.probs = [[],[],[]]
        self.colors = ['r','b','g','c','m','b']
        self.action_name = action_name

    def log(self, episode):
        for card in range(len(self.probs)):
            y = self.probs[card]
            prob = self.agent.greedyprob(card)
            y.append(prob)
        self.x.append(episode)

    def save(self, filename):
        plt.clf()
        for i in range(len(self.probs)):
            plt.plot(self.x, self.probs[i], label=NAMES[i], color=self.colors[i], marker = 'o', markevery=len(self.x)/30)
        plt.xlabel('Episodes')
        plt.ylabel('Probability of {0}'.format(self.action_name))
        plt.title('RL Agent Probabilistic Policy\n(Kuhn poker, {0})'.format(RUN_ID))
        plt.legend()
        plt.savefig('{0}_{1}.png'.format(filename, RUN_ID))

def plot_valuefn(agent, player):
    colors = ['r','b','g','c','m','b']
    for card in range(3):
        plt.clf()
        plt.plot([d[0] for d in agent.valuefn[card]], [d[1] for d in agent.valuefn[card]], label=NAMES[card], color=colors[card])
        if player == 1:
            plt.xlabel('Probability of betting')
        else:
            plt.xlabel('Probability of calling')
        plt.ylabel('Value')
        plt.title('Value Function for {0}-{1} actions\n(Kuhn poker, {2})'.format(player, NAMES[card], RUN_ID))
        plt.legend()
        plt.savefig('player{0}_{1}_{2}.png'.format(player, NAMES[card], RUN_ID))

if __name__ == "__main__":
    p1 = Agent(epsilon=0.1)
    p2 = Agent(epsilon=0.1)
    curve1 = LearningCurve(p1, "betting")
    curve2 = LearningCurve(p2, "calling")
    learning = 1
    learning_interval = 1000
    interleave = False
    for game in range(10000000):
        results = play_kuhn_hand(p1, p2)
        p1.reward(results[0])
        if results[2]:
            p2.reward(results[1])
        if game % 100 == 0:
            print game
            curve1.log(game)
            curve2.log(game)
        if interleave and game % learning_interval == 0:
            learning = (learning % 2) + 1
            if learning == 1:
                p2.exploit = True
                p1.exploit = False
            else:
                p1.exploit = True
                p2.exploit = False
    curve1.save('p1_policy')
    curve2.save('p2_policy')
    plot_valuefn(p1, 1)
    plot_valuefn(p2, 2)
