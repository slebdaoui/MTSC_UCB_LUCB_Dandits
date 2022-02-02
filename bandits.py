import numpy as np
import random
import math

from numpy.core.fromnumeric import mean

class oneBandit():
    # Reducing the hope of winning to complicate the problem 
    def __init__(self, cansino=None,mu=None):
        self.casino = cansino
        if mu is None:
            self._mu = random.random()/2 + 0.3 # moyenne de payoff de la machine >= 0.3
        else:
            self._mu = mu
        #self.random_generator =  random.Random() # Chaque machine dispose de son propre générateur de nombres aléatoires
        #self.random_generator.seed(self._mu)

    def playIt(self):
        return 1 if random.random() > self._mu else 0
    
class Casino():
    nb = 100
    def __init__(self, nb=100):
        self._machines = []
        self.nb
        self.last_rand = None
        for _ in range(nb):
            self._machines.append(oneBandit())
        self.best_mu = min(self._machines, key=lambda m: m._mu)
    def playMachine(self, i):
        return self._machines[i].playIt()
    # the probability to win is (1-m._mu)
    def realBestChoice(self): # On triche si on l'utilise. Doit être utilisé que pour calculer le regret
        return max(self._machines, key=lambda m: 1-m._mu)
    #Cette fonction renvoie la rewrard qui pouvait être attente si on jouait la meilleur machine
    def getMaxPossibleReward(self):
        return (self.last_rand > self.best_mu)
    def getNbM(self):
        return self.nb


stats_period  = 50

# Function implements greedy strategy
def greedy(h,Machine,nb_plays,return_stats=False):
    V = 0
    R_a     = np.zeros(nb_machine)
    nb_a    = np.zeros(nb_machine)
    nb_test_total = 0
    stats = None
    if return_stats:
        stats = []
    for i in range(Machine.getNbM()):
        if return_stats and i % stats_period//10 == 0:
            stats.append(R_a.sum())
        for j in range(nb_plays):
            R_a[i] += Machine.playMachine(i)
    best = np.argmax(R_a)
    for i in range(nb_plays*Machine.getNbM(),h):
        if return_stats and i % stats_period == 0:
            stats.append(R_a.sum())
        R_a[best] += Machine.playMachine(best)
    if return_stats :
        stats.append(R_a.sum())
        return  stats
    return R_a.sum()



def update_epsilon(epsilon):
    epsilon *= 0.98
    return max(epsilon,1e-10)

# We learn the wining average of each machine by exploration and exploitation.
# We start with epsilon = 1 (we have no experience), 
# and the value of epsilon decreases over time: epsilon 1 -> 1e-10.
def epsilon_greedy(h,CASINO,return_stats=False, fixed_epsilon=False, epsilon=1):
    nb_machines = CASINO.getNbM()

    nb_plays = 0
    stats = None
    R_a      = np.zeros(nb_machine) #rewards
    NB_tests = np.zeros(nb_machine) #nb tests per machine

    nb_test = nb_plays*nb_machine
    Total_reward = 0
    random_generator = random.Random()
    random_generator.seed(1)
    if return_stats:
        stats = []
    while nb_test < h:
        if return_stats and  nb_test % stats_period == 0:
            stats.append(Total_reward)
        r = 0
        m = -1
        if random_generator.random() < epsilon:
            m = random_generator.randint(0,nb_machines-1) #Choose random action #exploration 
        else :
            m = np.argmax(R_a) #exploitation
        
        r = CASINO.playMachine(m) # Play the choosen machine

        if NB_tests[m] == 0:
            R_a[m] = r
        else:
            R_a[m] = ((R_a[m]* NB_tests[m]) + r ) / (NB_tests[m] + 1)
        NB_tests[m] += 1
        
        Total_reward += r 
        if not fixed_epsilon:
            epsilon = update_epsilon(epsilon)
        nb_test += 1

    if return_stats :
        stats.append(Total_reward)
        return  stats
    return Total_reward

#-----------------------------------------------------------------------#
#-----------------------------   UCB   ---------------------------------#
#-----------------------------------------------------------------------#

def getMax_arg(R,nb_tests, nb_all, confidence,return_stats=False):
    M = np.zeros(len(R))
    for i in range(len(R)):
        if nb_tests[i] == 0:
            M[i] = math.inf
        else:
            M[i] = R[i] +  confidence * math.sqrt( 2 * math.log(nb_all) / nb_tests[i] )
    return np.argmax(M)

#Function implements  UCB
""" h: number of trials, CASINO: instance of Casino class """
def UCB(h,CASINO,confidence=0.2,return_stats=False):
    V = 0
    R_a     = np.zeros(nb_machine)
    nb_a    = np.zeros(nb_machine)
    nb_test_total = 0
    stats = None
    if return_stats:
        stats = []
    while nb_test_total < h:
        if return_stats and  nb_test_total % stats_period == 0:
            stats.append(V)
        m = getMax_arg(R_a,nb_a, nb_test_total, confidence)
        r = CASINO.playMachine(m)
        V += r
        R_a[m] = (nb_a[m] * R_a[m] + r) / (nb_a[m]+1)
        nb_a[m] += 1
        nb_test_total +=1
    if return_stats :
        stats.append(V)
        return  stats
    return V

#-----------------------------------------------------------------------#
#-----------------------------   LUCB   ---------------------------------#
#-----------------------------------------------------------------------#


class LUCB:
    def __init__(self,nb_machines,k,CASINO,epsilon=0.2, initial_credit=10000, confidence = 1):
        self.arms    = []#[i for i in range(nb_machines)]
        self.rewards = [0 for i in range(nb_machines)]
        self.total_reward = 0
        self.CASINO = CASINO
        self.nb_machines = nb_machines
        self.k = k
        self.epsilon = epsilon
        self.reward_mode = False
        self.t = 0 # iteration
        self.mean_reward = 0
        self.delta = 0.1
        self.cond_stop = -1
        self.stop = False
        self.initial_credit = initial_credit
        self.init()
        self.confidence  = confidence
    
    def init(self):
        for i in range(self.nb_machines):
            self.arms.append({})
            #self.arms[i] = 
            self.arms[i]['reward'] = 0
            self.arms[i]["lcb_i"]  = -math.inf
            self.arms[i]["ucb_i"]  =  math.inf
            self.arms[i]["id"] = i
            self.arms[i]["beta_i"] = 0
            self.arms[i]["t_i"] = 1
            
    def pull(self):
        #sort arms by reward
        sorted_arms = sorted(self.arms, key=lambda arm: arm['reward'])
        # divide them in two groups of k and n-k
        top = sorted_arms[-self.k :]
        bottom = sorted_arms[: self.nb_machines - self.k]

        # take arm from top with lowest lcb
        lcb_arm = sorted(top, key=lambda arm: arm['lcb_i'])[0]
        # if stop criteria met then stop
        ucb_arm = sorted(bottom, key=lambda arm: arm['ucb_i'])[-1]
        if abs(lcb_arm['lcb_i'] - ucb_arm['ucb_i']) < self.epsilon or self.initial_credit <= self.t:
            self.stop = True
            self.cond_stop = self.t + 1
            return False
        # Select random arm among lcb and ucb arms :
        a = random.choice([lcb_arm, ucb_arm])['id']
        self.play_m_searching(a)

        return True

    # Play arm and Update ucb_i / lcb_i / beta_i
    def play_m_searching(self,a):
        reward = self.CASINO.playMachine(a)
        self.total_reward += reward 
        # update counts
        self.t += 1
        self.arms[a]['t_i'] +=  1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.nb_machines
        # Update results for t_i
        self.arms[a]['reward'] = self.arms[a]['reward'] + (reward - self.arms[a]['reward']) / self.arms[a]['t_i']
        #self.arms[a]['beta_i'] = np.sqrt((1. / (2. * self.arms[a]['t_i'])) *
        #                                    np.log(1.25 * self.nb_machines * (self.t ** 4)) / self.delta)
        self.arms[a]['beta_i'] = self.confidence * np.sqrt((1. / (2. * self.arms[a]['t_i'])) *
                                            np.log(1. * self.t) / self.delta)
        #self.arms[a]['beta_i'] = math.sqrt( 2 * math.log(self.t) / self.arms[a]['t_i'] )
        self.arms[a]['ucb_i'] = self.arms[a]['reward'] + self.arms[a]['beta_i']
        self.arms[a]['lcb_i'] = self.arms[a]['reward'] - self.arms[a]['beta_i']
        return reward

    # Play arm and update just the reward mean
    def play_m_exp(self,a):
        reward = self.CASINO.playMachine(a)
        self.total_reward += reward 
        # update counts
        self.t += 1
        self.arms[a]['t_i'] +=  1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.nb_machines
        # Update results for t_i
        self.arms[a]['reward'] = self.arms[a]['reward'] + (reward - self.arms[a]['reward']) / self.arms[a]['t_i']
        return reward

    # Used
    def play_best_arm(self,display=False):
        a = sorted(self.arms, key=lambda arm: arm['reward'])[-1]["id"]
        r = self.play_m_exp(a)
        if display :
            print("Play arm:", a, " reward:",r)
        return r
    
    # Used
    def play_from_best_k_arms(self,display=False):
        sorted_arms = sorted(self.arms, key=lambda arm: arm['reward'])
        top = sorted_arms[-self.k :]
        a = sorted(top, key=lambda arm: arm['reward'])[-1]["id"]
        r = self.play_m_exp(a)
        if display :
            print("Play arm:", a, " reward:",r)
        return r
    
    def play_from_best_k_arms_reward(self,display=False):
        sorted_arms = sorted(self.arms, key=lambda arm: arm['reward'])
        top = sorted_arms[-self.k :]
        a = sorted(top, key=lambda arm: arm['ucb_i'])[-1]["id"]
        r = self.play_m_searching(a)
        if display :
            print("Play arm:", a, " reward:",r)
        return r
    # Used final 
    def play_from_best_k_arms_ucb(self,display=False):
        sorted_arms = sorted(self.arms, key=lambda arm: arm['ucb_i'])
        top = sorted_arms[-self.k :]
        a = sorted(self.arms, key=lambda arm: arm['reward'])[-1]["id"] #ucb_i
        r = self.play_m_searching(a)
        if display :
            print("Play arm:", a, " reward:",r)
        return r
    # Third call
    # Choose the arm with the largest reward among those of largest ucb 
    def play_all_best_from_k_ucb(self,All):
        while self.pull():
            continue
        rest = All - self.t 
        for i in range(rest):
            self.play_from_best_k_arms_ucb()
        return self.total_reward
    # Choose the arm with the largest ucb among those of largest reward
    def play_all_best_from_k_reward(self,All):
        while self.pull():
            continue
        rest = All - self.t 
        for i in range(rest):
            self.play_from_best_k_arms_ucb()
        return self.total_reward
    # First call
    # Choose the arm with the largest reward among those of largest ucb 
    def play_all_best_from_k(self,All):
        while self.pull():
            continue
        rest = All - self.t 
        for i in range(rest):
            self.play_from_best_k_arms()
        return self.total_reward
    
    #Second call 
    def play_all_best(self,All):
        while self.pull():
            continue
        rest = All - self.t 
        for i in range(rest):
            
            self.play_best_arm()
        return self.total_reward

#-----------------------------------------------------------------------------------#
#---------------------------- Tests de performance ---------------------------------#
#-----------------------------------------------------------------------------------#

initialCredits = 10000 # Nombre d'essais maximums

casino = Casino()
print(casino.realBestChoice()._mu)
nb_machine=100
epsilon = 6
k=10

if __name__ == '__main__':
    T = [[],[],[],[]]
    nb_tests = 20

    for i in range(nb_tests):
        T[0].append(greedy(initialCredits, casino,10))
        T[1].append(epsilon_greedy(initialCredits, casino))
        T[2].append(UCB(initialCredits, casino))

    print("Greedy  policy score        =", np.mean(T[0]))
    print("epsilon_greedy policy score =", np.mean(T[1]))
    print("UCB  policy score           =", np.mean(T[2]))
    print("Best from K : " , np.mean(L[0]))
    print("Best from all : " , np.mean(L[1]))
    print("UCB policy : " , np.mean(L[2]))
    print("Latest : " , np.mean(L[3]))
    print("reward : " , np.mean(L[4]))