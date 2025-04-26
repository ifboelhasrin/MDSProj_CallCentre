import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import matplotlib
import time
from tqdm import tqdm

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
SERVICE_COMPLEX_1 = 3  # Fixed from the parameters (was SERVICE_COMPLEX_2)
## Agent 2
SERVICE_SIMPLE_2 = 6
SERVICE_COMPLEX_2 = 5

# Discount factor for future rewards
DISCOUNT = 0.95

# Upper bound for poisson distribution
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
poisson_cache = dict()

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

# Agent state: 0 = idle, 1 = serving simple query, 2 = serving complex query
AGENT_IDLE = 0
AGENT_SIMPLE = 1
AGENT_COMPLEX = 2

# Action: 0 = reject, 1 = route to queue 1, 2 = route to queue 2
ACTION_REJECT = 0
ACTION_QUEUE1 = 1
ACTION_QUEUE2 = 2

# Query types
QUERY_SIMPLE = 0
QUERY_COMPLEX = 1

class CallCenterMDP:
    def __init__(self):
        # Define state space dimensions
        # State = (q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status)
        self.state_dims = (MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1, MAX_QUEUE_SIZE + 1, 3, 3)
        
        # Total number of states
        self.num_states = np.prod(self.state_dims)
        
        # Number of actions
        self.num_actions = 3  # Reject, Queue1, Queue2
        
        # Initialize value function
        self.value_function = np.zeros(self.state_dims)
        
        # Initialize policy
        self.policy_simple = np.zeros(self.state_dims, dtype=int)
        self.policy_complex = np.zeros(self.state_dims, dtype=int)
        
        # Probability of query type
        self.p_simple = ARRIVAL_SIMPLE / (ARRIVAL_SIMPLE + ARRIVAL_COMPLEX)
        self.p_complex = ARRIVAL_COMPLEX / (ARRIVAL_SIMPLE + ARRIVAL_COMPLEX)
        
        # Uniformization constant (for CTMDP conversion)
        self.uniformization_rate = ARRIVAL_SIMPLE + ARRIVAL_COMPLEX + SERVICE_SIMPLE_1 + SERVICE_COMPLEX_1 + SERVICE_SIMPLE_2 + SERVICE_COMPLEX_2
        
    def is_valid_state(self, state):
        q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
        
        # Check queue limits
        q1_total = q1_simple + q1_complex
        q2_total = q2_simple + q2_complex
        
        if q1_total > MAX_QUEUE_SIZE or q2_total > MAX_QUEUE_SIZE:
            return False
        
        # Check agent status
        if agent1_status not in [AGENT_IDLE, AGENT_SIMPLE, AGENT_COMPLEX]:
            return False
        if agent2_status not in [AGENT_IDLE, AGENT_SIMPLE, AGENT_COMPLEX]:
            return False
        
        # If agent is busy, there should be no items in queue of the same type
        if agent1_status == AGENT_SIMPLE and q1_simple > 0:
            return False
        if agent1_status == AGENT_COMPLEX and q1_complex > 0:
            return False
        if agent2_status == AGENT_SIMPLE and q2_simple > 0:
            return False
        if agent2_status == AGENT_COMPLEX and q2_complex > 0:
            return False
        
        return True
    
    def get_waiting_time(self, state, queue, query_type):
        """Calculate expected waiting time for a new query in the specified queue"""
        q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
        
        if queue == 1:
            # Waiting time in queue 1
            if query_type == QUERY_SIMPLE:
                # For a simple query in queue 1
                waiting_time = q1_simple / SERVICE_SIMPLE_1
                if agent1_status == AGENT_SIMPLE:
                    waiting_time += 1 / SERVICE_SIMPLE_1  # Remaining service time
                elif agent1_status == AGENT_COMPLEX:
                    waiting_time += 1 / SERVICE_COMPLEX_1  # Remaining service time
                return waiting_time
            else:
                # For a complex query in queue 1
                waiting_time = q1_complex / SERVICE_COMPLEX_1
                if agent1_status == AGENT_SIMPLE:
                    waiting_time += 1 / SERVICE_SIMPLE_1  # Remaining service time
                elif agent1_status == AGENT_COMPLEX:
                    waiting_time += 1 / SERVICE_COMPLEX_1  # Remaining service time
                return waiting_time
        else:
            # Waiting time in queue 2
            if query_type == QUERY_SIMPLE:
                # For a simple query in queue 2
                waiting_time = q2_simple / SERVICE_SIMPLE_2
                if agent2_status == AGENT_SIMPLE:
                    waiting_time += 1 / SERVICE_SIMPLE_2  # Remaining service time
                elif agent2_status == AGENT_COMPLEX:
                    waiting_time += 1 / SERVICE_COMPLEX_2  # Remaining service time
                return waiting_time
            else:
                # For a complex query in queue 2
                waiting_time = q2_complex / SERVICE_COMPLEX_2
                if agent2_status == AGENT_SIMPLE:
                    waiting_time += 1 / SERVICE_SIMPLE_2  # Remaining service time
                elif agent2_status == AGENT_COMPLEX:
                    waiting_time += 1 / SERVICE_COMPLEX_2  # Remaining service time
                return waiting_time
    
    def reward(self, state, action, query_type):
        """Calculate reward (negative cost) for an action in a state"""
        q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
        
        # Calculate total items in each queue
        q1_total = q1_simple + q1_complex
        q2_total = q2_simple + q2_complex
        
        # If action is reject
        if action == ACTION_REJECT:
            return -100  # High penalty for rejection
        
        # If action is route to queue 1
        elif action == ACTION_QUEUE1:
            # Check if queue 1 is full
            if q1_total >= MAX_QUEUE_SIZE:
                return -100  # Queue is full, cannot route here
            # Return negative waiting time as cost
            return -self.get_waiting_time(state, 1, query_type)
        
        # If action is route to queue 2
        elif action == ACTION_QUEUE2:
            # Check if queue 2 is full
            if q2_total >= MAX_QUEUE_SIZE:
                return -100  # Queue is full, cannot route here
            # Return negative waiting time as cost
            return -self.get_waiting_time(state, 2, query_type)
        
        return 0
    
    def next_state(self, state, action, query_type):
        """Get the next state after taking an action"""
        q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
        
        # Make a copy of the current state
        next_q1_simple, next_q1_complex = q1_simple, q1_complex
        next_q2_simple, next_q2_complex = q2_simple, q2_complex
        next_agent1_status, next_agent2_status = agent1_status, agent2_status
        
        # Apply action
        if action == ACTION_QUEUE1:
            if query_type == QUERY_SIMPLE:
                # Add simple query to queue 1
                if agent1_status == AGENT_IDLE:
                    next_agent1_status = AGENT_SIMPLE
                else:
                    next_q1_simple += 1
            else:
                # Add complex query to queue 1
                if agent1_status == AGENT_IDLE:
                    next_agent1_status = AGENT_COMPLEX
                else:
                    next_q1_complex += 1
        
        elif action == ACTION_QUEUE2:
            if query_type == QUERY_SIMPLE:
                # Add simple query to queue 2
                if agent2_status == AGENT_IDLE:
                    next_agent2_status = AGENT_SIMPLE
                else:
                    next_q2_simple += 1
            else:
                # Add complex query to queue 2
                if agent2_status == AGENT_IDLE:
                    next_agent2_status = AGENT_COMPLEX
                else:
                    next_q2_complex += 1
        
        return (next_q1_simple, next_q1_complex, next_q2_simple, next_q2_complex, next_agent1_status, next_agent2_status)
    
    def value_iteration(self, max_iterations=1000, epsilon=1e-4):
        """Solve the MDP using value iteration"""
        start_time = time.time()
        
        # Initialize value function
        V = np.zeros(self.state_dims)
        
        # Initialize policies
        policy_simple = np.zeros(self.state_dims, dtype=int)
        policy_complex = np.zeros(self.state_dims, dtype=int)
        
        # Value iteration
        for iteration in tqdm(range(max_iterations)):
            delta = 0
            V_new = np.zeros(self.state_dims)
            
            # Iterate over all states
            for q1_simple in range(MAX_QUEUE_SIZE + 1):
                for q1_complex in range(MAX_QUEUE_SIZE + 1):
                    # Skip invalid queue combinations
                    if q1_simple + q1_complex > MAX_QUEUE_SIZE:
                        continue
                    
                    for q2_simple in range(MAX_QUEUE_SIZE + 1):
                        for q2_complex in range(MAX_QUEUE_SIZE + 1):
                            # Skip invalid queue combinations
                            if q2_simple + q2_complex > MAX_QUEUE_SIZE:
                                continue
                            
                            for agent1_status in range(3):
                                for agent2_status in range(3):
                                    state = (q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status)
                                    
                                    # Skip invalid states
                                    if not self.is_valid_state(state):
                                        continue
                                    
                                    # Process simple query
                                    max_value_simple = float('-inf')
                                    best_action_simple = ACTION_REJECT
                                    
                                    for action in range(self.num_actions):
                                        # Calculate reward and next state
                                        r = self.reward(state, action, QUERY_SIMPLE)
                                        next_s = self.next_state(state, action, QUERY_SIMPLE)
                                        
                                        # Calculate value
                                        value = r + DISCOUNT * V[next_s]
                                        
                                        if value > max_value_simple:
                                            max_value_simple = value
                                            best_action_simple = action
                                    
                                    # Process complex query
                                    max_value_complex = float('-inf')
                                    best_action_complex = ACTION_REJECT
                                    
                                    for action in range(self.num_actions):
                                        # Calculate reward and next state
                                        r = self.reward(state, action, QUERY_COMPLEX)
                                        next_s = self.next_state(state, action, QUERY_COMPLEX)
                                        
                                        # Calculate value
                                        value = r + DISCOUNT * V[next_s]
                                        
                                        if value > max_value_complex:
                                            max_value_complex = value
                                            best_action_complex = action
                                    
                                    # Calculate expected value
                                    expected_value = self.p_simple * max_value_simple + self.p_complex * max_value_complex
                                    
                                    # Update value function
                                    V_new[state] = expected_value
                                    
                                    # Update policies
                                    policy_simple[state] = best_action_simple
                                    policy_complex[state] = best_action_complex
                                    
                                    # Track maximum change
                                    delta = max(delta, abs(V_new[state] - V[state]))
            
            # Update value function
            V = V_new.copy()
            
            # Check for convergence
            if delta < epsilon:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Save final value function and policy
        self.value_function = V
        self.policy_simple = policy_simple
        self.policy_complex = policy_complex
        
        elapsed_time = time.time() - start_time
        print(f"Value iteration completed in {elapsed_time:.2f} seconds")
    
    def simulate(self, num_steps=10000):
        """Simulate the call center using the learned policy"""
        # Initialize statistics
        waiting_times = []
        rejection_count = 0
        total_arrivals = 0
        
        # Initialize state
        state = (0, 0, 0, 0, AGENT_IDLE, AGENT_IDLE)
        
        # Simulate
        for step in range(num_steps):
            # Randomly determine query type
            query_type = QUERY_SIMPLE if np.random.random() < self.p_simple else QUERY_COMPLEX
            
            # Get action from policy
            if query_type == QUERY_SIMPLE:
                action = self.policy_simple[state]
            else:
                action = self.policy_complex[state]
            
            # Track arrivals and rejections
            total_arrivals += 1
            if action == ACTION_REJECT:
                rejection_count += 1
                continue
            
            # Calculate waiting time
            q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
            
            if action == ACTION_QUEUE1:
                waiting_time = self.get_waiting_time(state, 1, query_type)
            else:  # ACTION_QUEUE2
                waiting_time = self.get_waiting_time(state, 2, query_type)
            
            waiting_times.append(waiting_time)
            
            # Update state
            state = self.next_state(state, action, query_type)
            
            # Simulate service completions
            q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status = state
            
            # Agent 1 service completion
            if agent1_status == AGENT_SIMPLE:
                if np.random.random() < SERVICE_SIMPLE_1 / self.uniformization_rate:
                    agent1_status = AGENT_IDLE
            elif agent1_status == AGENT_COMPLEX:
                if np.random.random() < SERVICE_COMPLEX_1 / self.uniformization_rate:
                    agent1_status = AGENT_IDLE
            
            # Agent 2 service completion
            if agent2_status == AGENT_SIMPLE:
                if np.random.random() < SERVICE_SIMPLE_2 / self.uniformization_rate:
                    agent2_status = AGENT_IDLE
            elif agent2_status == AGENT_COMPLEX:
                if np.random.random() < SERVICE_COMPLEX_2 / self.uniformization_rate:
                    agent2_status = AGENT_IDLE
            
            # If agent becomes idle, pick next task from queue if available
            if agent1_status == AGENT_IDLE:
                if q1_simple > 0:
                    q1_simple -= 1
                    agent1_status = AGENT_SIMPLE
                elif q1_complex > 0:
                    q1_complex -= 1
                    agent1_status = AGENT_COMPLEX
            
            if agent2_status == AGENT_IDLE:
                if q2_simple > 0:
                    q2_simple -= 1
                    agent2_status = AGENT_SIMPLE
                elif q2_complex > 0:
                    q2_complex -= 1
                    agent2_status = AGENT_COMPLEX
            
            # Update state
            state = (q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status)
        
        # Calculate and return statistics
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        rejection_rate = rejection_count / total_arrivals if total_arrivals > 0 else 0
        
        print(f"Average waiting time: {avg_waiting_time:.2f}")
        print(f"Rejection rate: {rejection_rate:.2%}")
        
        return waiting_times, rejection_rate

    def analyze_policy(self):
        """Analyze the learned policy"""
        # Initialize counts
        simple_to_queue1 = 0
        simple_to_queue2 = 0
        complex_to_queue1 = 0
        complex_to_queue2 = 0
        simple_rejected = 0
        complex_rejected = 0
        
        total_states = 0
        
        # Count routing decisions for each state
        for q1_simple in range(MAX_QUEUE_SIZE + 1):
            for q1_complex in range(MAX_QUEUE_SIZE + 1):
                if q1_simple + q1_complex > MAX_QUEUE_SIZE:
                    continue
                
                for q2_simple in range(MAX_QUEUE_SIZE + 1):
                    for q2_complex in range(MAX_QUEUE_SIZE + 1):
                        if q2_simple + q2_complex > MAX_QUEUE_SIZE:
                            continue
                        
                        for agent1_status in range(3):
                            for agent2_status in range(3):
                                state = (q1_simple, q1_complex, q2_simple, q2_complex, agent1_status, agent2_status)
                                
                                if not self.is_valid_state(state):
                                    continue
                                
                                total_states += 1
                                
                                # Analyze simple query routing
                                action = self.policy_simple[state]
                                if action == ACTION_QUEUE1:
                                    simple_to_queue1 += 1
                                elif action == ACTION_QUEUE2:
                                    simple_to_queue2 += 1
                                else:
                                    simple_rejected += 1
                                
                                # Analyze complex query routing
                                action = self.policy_complex[state]
                                if action == ACTION_QUEUE1:
                                    complex_to_queue1 += 1
                                elif action == ACTION_QUEUE2:
                                    complex_to_queue2 += 1
                                else:
                                    complex_rejected += 1
        
        # Calculate percentages
        print(f"Policy analysis (over {total_states} valid states):")
        print(f"Simple queries routed to Queue 1: {simple_to_queue1 / total_states:.2%}")
        print(f"Simple queries routed to Queue 2: {simple_to_queue2 / total_states:.2%}")
        print(f"Simple queries rejected: {simple_rejected / total_states:.2%}")
        print(f"Complex queries routed to Queue 1: {complex_to_queue1 / total_states:.2%}")
        print(f"Complex queries routed to Queue 2: {complex_to_queue2 / total_states:.2%}")
        print(f"Complex queries rejected: {complex_rejected / total_states:.2%}")
        
        # Analyze specific scenarios
        print("\nAnalysis of specific scenarios:")
        
        # Empty system
        empty_state = (0, 0, 0, 0, AGENT_IDLE, AGENT_IDLE)
        print("Empty system:")
        print(f"  Simple query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_simple[empty_state]]}")
        print(f"  Complex query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_complex[empty_state]]}")
        
        # Only Queue 1 has space
        queue1_only = (0, 0, MAX_QUEUE_SIZE, 0, AGENT_IDLE, AGENT_IDLE)
        if self.is_valid_state(queue1_only):
            print("Only Queue 1 has space:")
            print(f"  Simple query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_simple[queue1_only]]}")
            print(f"  Complex query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_complex[queue1_only]]}")
        
        # Only Queue 2 has space
        queue2_only = (MAX_QUEUE_SIZE, 0, 0, 0, AGENT_IDLE, AGENT_IDLE)
        if self.is_valid_state(queue2_only):
            print("Only Queue 2 has space:")
            print(f"  Simple query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_simple[queue2_only]]}")
            print(f"  Complex query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_complex[queue2_only]]}")
        
        # Equal load
        equal_load = (5, 5, 5, 5, AGENT_IDLE, AGENT_IDLE)
        if self.is_valid_state(equal_load):
            print("Equal load (5 of each query type in each queue):")
            print(f"  Simple query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_simple[equal_load]]}")
            print(f"  Complex query -> {['Reject', 'Queue 1', 'Queue 2'][self.policy_complex[equal_load]]}")

# Run the model
def main():
    print("Initializing Call Center MDP...")
    mdp = CallCenterMDP()
    
    print("Running value iteration...")
    mdp.value_iteration(max_iterations=100)
    
    print("\nAnalyzing policy...")
    mdp.analyze_policy()
    
    print("\nRunning simulation...")
    waiting_times, rejection_rate = mdp.simulate(num_steps=10000)
    
    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.hist(waiting_times, bins=50, alpha=0.7)
    plt.title('Distribution of Waiting Times')
    plt.xlabel('Waiting Time')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('waiting_times_distribution.png')
    plt.show()
    
    print("\nSimulation results:")
    print(f"Average waiting time: {np.mean(waiting_times):.2f}")
    print(f"Median waiting time: {np.median(waiting_times):.2f}")
    print(f"Maximum waiting time: {np.max(waiting_times):.2f}")
    print(f"Rejection rate: {rejection_rate:.2%}")

if __name__ == "__main__":
    main()