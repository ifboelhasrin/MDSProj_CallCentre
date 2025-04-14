import numpy as np
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from scipy.stats import poisson

matplotlib.use('Agg')

# Global variables

# Time horizon
TIME_HORIZON = 1000

# Queue size
MAX_QUEUE_SIZE = 20

# Arrival average of customers per unit time (Poisson process)
ARRIVAL_SIMPLE = 8
ARRIVAL_COMPLEX = 5

# Service average of customers per unit time (Exponential distribution)
## Agent 1
SERVICE_SIMPLE_1 = 8
SERVICE_COMPLEX_2 = 3
## Agent 2
SERVICE_SIMPLE_2 = 6
SERVICE_COMPLEX_2 = 5

# Discount factor for future rewards
DISCOUNT = 0.95

# State space: (queue 1 size, queue 2 size, call type)
STATE_SPACE = (MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1, 2)
# State space size
STATE_SIZE = np.prod(STATE_SPACE)

# Call type: 0 = simple, 1 = complex
CALL_TYPE = [0, 1]

# Actions: 0 = route to queue 1, 1 = route to queue 2
ACTIONS = [0, 1]

# Upper bound for poisson distribution
# Because the poisson distribution may produce a very large number of customers, which is not realistic
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
poisson_cache = dict()

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


class CallCentreMDP:
    def __init__(self):
        
        # Initialise value function
        self.value = np.zeros(STATE_SIZE)
        
        # Initialise policy
        self.policy = np.zeros(STATE_SIZE, dtype=int)
        
    def expected_reward(self, state, action, state_value):
        
        # Initialise expected reward
        rewards = 0.0
        
        # Reward should be negative of waiting time
        # Plus expected future reward 