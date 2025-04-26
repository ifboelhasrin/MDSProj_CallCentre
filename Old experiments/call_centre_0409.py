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
POISSON_UPPER_BOUND = 15

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
        
        # Initialise policy matrices
        # One for each call types
        self.policy = np.zeros((len(ACTIONS), MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1), dtype=int)
        
        self.state_values = np.array(self.state_values)
                    
    # def expected_waiting_time(self, state, action):
        
    #     # Get state values
    #     q1, q2, call_type = state
        
    #     waiting_time = 0.0
        
    #     # If action is to route to queue 1
    #     if action == 0:
    #         # If queue 1 is empty, waiting time is 0
    #         if q1 == 0:
    #             return waiting_time
            
    #         # If queue 1 is not empty, calculate waiting time
    #         waiting_time = (q1 - 1) / SERVICE_SIMPLE_1 if call_type == 0 else (q1 - 1) / SERVICE_COMPLEX_2
    
    def expected_routing(self, state, action, state_value):
        
        # Initialise expected reward
        rewards = 0.0
        
        # Number of customers in queue 1 and 2
        NUM_CUSTOMERS_QUEUE_1 = min(state[0], MAX_QUEUE_SIZE)
        NUM_CUSTOMERS_QUEUE_2 = min(state[1], MAX_QUEUE_SIZE)
        
        for calls_simple in range(POISSON_UPPER_BOUND):
            for calls_complex in range(POISSON_UPPER_BOUND):
                
                # Get poisson probability
                prob = poisson_probability(calls_simple, ARRIVAL_SIMPLE) * \
                    poisson_probability(calls_complex, ARRIVAL_COMPLEX)
                    
                
                
                
                
                
                
        
        # Reward should be negative of waiting time
        # Plus expected future reward 
        
    def value_iteration():
        
        # Initialise value function
        value = np.zeros(STATE_SIZE)
        
        # Initialise policy matrices
        # One for each call types
        policy = np.zeros((len(ACTIONS), MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1), dtype=int)
        
        # Value iteration
        for _ in tqdm(range(TIME_HORIZON)):
            
            # For each state
            for q1 in range(MAX_QUEUE_SIZE + 1):
                for q2 in range(MAX_QUEUE_SIZE + 1):
                    
                    # Get state index
                    state_index = q1 * (MAX_QUEUE_SIZE + 1) + q2
                    
                    # For each action
                    for action in ACTIONS:
                        
                        # Get expected reward and future reward
                        rewards = expected_routing((q1, q2, CALL_TYPE[0]), action, value)
                        
                        # Update value function
                        if rewards > value[state_index]:
                            value[state_index] = rewards
                            policy[action, q1, q2] = action
        
        return policy, value