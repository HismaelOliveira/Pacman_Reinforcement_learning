from game import *

import random,util,math,time
import util
import pandas as pd 

def closestFood(pos, food, walls):
    matrix = [(pos[0], pos[1], 0)]
    neigh = set()
    while matrix:

        pos_x, pos_y, dist = matrix.pop(0)
        if (pos_x, pos_y) not in neigh:
            neigh.add((pos_x, pos_y))

            if food[pos_x][pos_y]:
                return dist

            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                matrix.append((nbr_x, nbr_y, dist+1))
    return None

class RLAgent():

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.QValues = util.Counter()
        self.weights = util.Counter() 
        self.scores = []
        self.actions = []
        self.importance = []
        self.rewards = []
        self.rewards_window = []

    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0    
            self.alpha = 0.0     

    def computeValueFromQValues(self, state):
        values = [self.getQValue(state, action) for action in state.getLegalActions()]
        if (values):
            return max(values)
        else:
            return 0.0 

    def computeActionFromQValues(self, state):
        legal_actions = state.getLegalActions()

        value = self.computeValueFromQValues(state)
        for action in legal_actions:
            if (value == self.getQValue(state, action)):
                return action

    def getAction(self, state):
        legalActions = state.getLegalActions()
        action = None

        if (util.flipCoin(self.epsilon)):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        self.lastState = state
        self.lastAction = action
        self.numActions += 1

        return action

    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsule = state.getCapsules()

        features = util.Counter()

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_2x, next_2y = int(x + 2*dx), int(y + 2*dy)

        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        features["#-of-ghosts-2-step-away"] = sum((next_2x, next_2y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features

    def getQValue(self, state, action):
        features = self.getFeatures(state,action)
        QValue = 0.0

        for feature in features:
            QValue += self.weights[feature] * features[feature]

        return QValue

    def update(self, state, action, nextState, reward):
        QValue = 0
        difference = reward + (self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        features = self.getFeatures(state, action)

        for feature in features:
            self.weights[feature] += self.alpha * features[feature] * difference

    def registerInitialState(self, state):
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0
        self.numActions = 0

        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state):
        deltaReward = state.getScore() - self.lastState.getScore()
        self.episodeRewards += deltaReward   
        self.update(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 5
        if self.episodesSoFar % NUM_EPS_UPDATE == 0: 
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            
            trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            self.lastWindowAccumRewards = 0.0
            self.rewards_window.append(trainAvg)
            
        self.scores.append(state.getScore())
        self.actions.append(self.numActions)
        self.importance.append(self.weights)
        self.rewards.append(self.episodeRewards)

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Process is Done'
            print ('%s\n%s' % (msg,'-' * len(msg)))
            print ('\t%d training episodes ' % (self.numTraining))
            print ('\tAverage Rewards over all training: %.2f' % (
                    trainAvg))

            data = pd.DataFrame()
            data['Scores'] = self.scores 
            data['Actions'] = self.actions
            data['Weights'] = self.importance
            data['Rewards'] = self.rewards

            rewards = pd.DataFrame()
            rewards['rewards'] = self.rewards_window
            
            data.to_excel('data_pacman.xlsx')
            rewards.to_excel('rewards_pacman.xlsx')