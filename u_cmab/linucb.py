from abc import ABC, abstractmethod
from numpy.linalg import inv
import numpy as np

class MAB(ABC):
    
    @abstractmethod
    def play(self, tround, context):
        # Current round of t (for my implementations average mean reward array 
        # at round t is passed to this function instead of tround itself)
        self.tround = tround
        # Context: features of contextual bandits
        self.context = context
        # choose an arm which yields maximum value of average mean reward, tie breaking randomly
        chosen_arm = np.random.choice(np.where(self.tround==max(self.tround))[0])
        return chosen_arm
        pass
        
    
    @abstractmethod
    def update(self, arm, reward, context):
        # get the chosen arm
        self.arm = arm 
        # get the context (may be None)
        self.context = context
        # update the overall step of the model
        self.step_n += 1
        # update the step of individual arms
        self.step_arm[self.arm] += 1
        # update average mean reward of each arm
        self.AM_reward[self.arm] = ((self.step_arm[self.arm] - 1) / float(self.step_arm[self.arm]) 
        * self.AM_reward[self.arm] + (1 / float(self.step_arm[self.arm])) * reward)
        return
        pass


def offlineEvaluate(mab, rewards, contexts, tau, nrounds=None, u_cmab=1):
    # array to contain chosen arms in offline mode
    chosen_arms = np.zeros(nrounds)
    # rewards of each chosen arm
    reward_arms = np.zeros(nrounds)
    # cumulative reward at each iteration
    cumulative_reward = np.zeros(nrounds)
    # initialize tround to zero
    T = 0
    # initialize overall cumulative reward to zero
    G = 0
    # History or memory of offline evaluator
    history = []
    # play once and get the initial action
    action = mab.play(T, contexts[0,:])
    
    #===============================
    #    MAIN LOOP ...
    #===============================
    for i in range(np.shape(contexts)[0]):
        action = mab.play(T, contexts[i,:])
        if T<nrounds:
            # append the current context of chosen arm to the previous history (list)
            history.append(contexts[i,:])
            # get the reward of chosen arm at round T
            reward_arms[T] = rewards[i][action] - action * tau * u_cmab
            # the returned action is between 1-10, setting to python encoding ==> 0-9
            mab.update(action, rewards[i][action] - action * tau * u_cmab, contexts[i,:])
            # update overall cumulative reward
            G += rewards[i][action]
            # update cumulative reward of round T 
            cumulative_reward[T] = G
            # store chosen arm at round T
            chosen_arms[T] = action
            T +=1
        else:
            # if desired tround ends, terminate the loop
            break
    return reward_arms, chosen_arms, cumulative_reward


class LinUCB(MAB):
    
    def __init__(self, narms, ndims, alpha):
        # Set number of arms
        self.narms = narms
        # Number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.alpha = alpha
        # Instantiate A as a ndims×ndims matrix for each arm
        self.A = np.zeros((self.narms, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((narms, self.ndims, 1))
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.narms):
            self.A[arm] = np.eye(self.ndims)
        
        super().__init__()
        return
        
    def play(self, tround, context):
        # gains per each arm
        p_t = np.zeros(self.ndims)
        context = np.reshape(context, (self.narms, self.ndims))
        
        #===============================
        #    MAIN LOOP ...
        #===============================
        for i in range(self.narms):
            # initialize theta hat
            self.theta = inv(self.A[i]).dot(self.b[i])
            # get context of each arm from flattened vector of length 100
            cntx = context[i]
            # get gain reward of each arm
            p_t[i] = self.theta.T.dot(cntx
                ) + self.alpha * np.sqrt(
            cntx.dot(inv(self.A[i]).dot(cntx)))
        action = np.random.choice(np.where(p_t==max(p_t))[0])
        # np.argmax returns values 0-9, we want to compare with arm indices in dataset which are 1-10
        # Hence, add 1 to action before returning
        return action
        
    
    def update(self, arm, reward, context):
        context = np.reshape(context, (self.narms, self.ndims))
        
        self.A[arm] = self.A[arm] + np.outer(context[arm],context[arm])
        self.b[arm] = np.add(self.b[arm].T, context[arm]*reward).reshape(self.ndims,1)
        return
