# Mathematical Model for Call Center Routing using MDP

## System Components

1. **Agents**: Two agents (Agent 1 and Agent 2) with different service capabilities
2. **Query Types**: Two types - Simple (S) and Complex (C)
3. **Queues**: Two queues, one for each agent, with maximum capacities

## Parameters

- $\lambda_S$: Arrival rate of simple queries (Poisson distributed)
- $\lambda_C$: Arrival rate of complex queries (Poisson distributed)
- $\mu_{1S}$: Service rate of Agent 1 for simple queries (Exponential distribution)
- $\mu_{1C}$: Service rate of Agent 1 for complex queries (Exponential distribution)
- $\mu_{2S}$: Service rate of Agent 2 for simple queries (Exponential distribution)
- $\mu_{2C}$: Service rate of Agent 2 for complex queries (Exponential distribution)
- $Q_1^{max}$: Maximum capacity of queue 1
- $Q_2^{max}$: Maximum capacity of queue 2

## Assumptions

1. Agent 1 is better at handling complex queries: $\mu_{1C} > \mu_{2C}$
2. Agent 2 is better at handling simple queries: $\mu_{2S} > \mu_{1S}$
3. Simple queries are more common: $\lambda_S > \lambda_C$
4. No call abandonment
5. Call arrivals and service times are memoryless

## State Space

The state of the system can be represented as:
$s = (q_1^S, q_1^C, q_2^S, q_2^C, b_1, b_2)$

Where:
- $q_1^S$: Number of simple queries in queue 1
- $q_1^C$: Number of complex queries in queue 1
- $q_2^S$: Number of simple queries in queue 2
- $q_2^C$: Number of complex queries in queue 2
- $b_1 \in \{0, S, C\}$: Status of Agent 1 (0: idle, S: serving simple query, C: serving complex query)
- $b_2 \in \{0, S, C\}$: Status of Agent 2 (0: idle, S: serving simple query, C: serving complex query)

Constraints:
- $0 \leq q_1^S + q_1^C \leq Q_1^{max}$
- $0 \leq q_2^S + q_2^C \leq Q_2^{max}$

## Action Space

For each arriving call of type $t \in \{S, C\}$, the actions are:
- $a = 1$: Route to Queue 1
- $a = 2$: Route to Queue 2
- $a = 0$: Reject the call (only when both queues are full)

## Transition Probabilities

The state transitions depend on:
1. New call arrivals
2. Service completions by agents
3. Routing decisions

For a small time interval $\Delta t$, the transition probabilities are:

1. **New call arrival of type $S$ with probability $\lambda_S \Delta t$**:
   - If action $a = 1$ and $q_1^S + q_1^C < Q_1^{max}$, increment $q_1^S$ by 1
   - If action $a = 2$ and $q_2^S + q_2^C < Q_2^{max}$, increment $q_2^S$ by 1
   - Otherwise (queues full), call is rejected

2. **New call arrival of type $C$ with probability $\lambda_C \Delta t$**:
   - If action $a = 1$ and $q_1^S + q_1^C < Q_1^{max}$, increment $q_1^C$ by 1
   - If action $a = 2$ and $q_2^S + q_2^C < Q_2^{max}$, increment $q_2^C$ by 1
   - Otherwise (queues full), call is rejected

3. **Service completion by Agent 1**:
   - If $b_1 = S$, with probability $\mu_{1S} \Delta t$, set $b_1 = 0$
   - If $b_1 = C$, with probability $\mu_{1C} \Delta t$, set $b_1 = 0$

4. **Service completion by Agent 2**:
   - If $b_2 = S$, with probability $\mu_{2S} \Delta t$, set $b_2 = 0$
   - If $b_2 = C$, with probability $\mu_{2C} \Delta t$, set $b_2 = 0$

5. **Agent picks next call from queue**:
   - If $b_1 = 0$ and $q_1^S + q_1^C > 0$, agent 1 takes next call (prioritizing based on policy)
   - If $b_2 = 0$ and $q_2^S + q_2^C > 0$, agent 2 takes next call (prioritizing based on policy)

## Reward (Cost) Function

The reward function is negative (cost function) as we aim to minimize waiting time:

For an action $a$ in state $s$ when a call of type $t$ arrives:

- If $a = 0$ (reject): $R(s, a, t) = -P$ (penalty for rejection)
- If $a = 1$ (route to Queue 1): $R(s, a, t) = -W_1(s, t)$ (expected waiting time in Queue 1)
- If $a = 2$ (route to Queue 2): $R(s, a, t) = -W_2(s, t)$ (expected waiting time in Queue 2)

Where $W_i(s, t)$ represents the expected waiting time for a call of type $t$ assigned to queue $i$ in state $s$:

$W_1(s, S) = \frac{q_1^S}{\mu_{1S}} + \frac{q_1^C}{\mu_{1C}} + \text{remaining service time of current call being served by Agent 1}$

$W_1(s, C) = \frac{q_1^S}{\mu_{1S}} + \frac{q_1^C}{\mu_{1C}} + \text{remaining service time of current call being served by Agent 1}$

$W_2(s, S) = \frac{q_2^S}{\mu_{2S}} + \frac{q_2^C}{\mu_{2C}} + \text{remaining service time of current call being served by Agent 2}$

$W_2(s, C) = \frac{q_2^S}{\mu_{2S}} + \frac{q_2^C}{\mu_{2C}} + \text{remaining service time of current call being served by Agent 2}$

## Bellman Equation

The optimal value function $V^*(s)$ satisfies the Bellman equation:

$V^*(s) = \sum_{t \in \{S, C\}} p(t) \max_a \left[ R(s, a, t) + \gamma \sum_{s'} P(s'|s, a, t) V^*(s') \right]$

Where:
- $p(S) = \frac{\lambda_S}{\lambda_S + \lambda_C}$ and $p(C) = \frac{\lambda_C}{\lambda_S + \lambda_C}$ are the probabilities of arrival types
- $\gamma$ is the discount factor (close to 1 for this problem)
- $P(s'|s, a, t)$ is the transition probability from state $s$ to $s'$ when action $a$ is taken for a call of type $t$

## Optimal Policy

The optimal policy $\pi^*(s, t)$ specifies the best action for each state $s$ and arrival type $t$:

$\pi^*(s, t) = \arg\max_a \left[ R(s, a, t) + \gamma \sum_{s'} P(s'|s, a, t) V^*(s') \right]$

## Dynamic Programming Solution

To solve this MDP, we can use value iteration:

1. Initialize $V_0(s) = 0$ for all states $s$
2. For $k = 0, 1, 2, \ldots$ until convergence:
   $V_{k+1}(s) = \sum_{t \in \{S, C\}} p(t) \max_a \left[ R(s, a, t) + \gamma \sum_{s'} P(s'|s, a, t) V_k(s') \right]$
3. Extract the optimal policy:
   $\pi^*(s, t) = \arg\max_a \left[ R(s, a, t) + \gamma \sum_{s'} P(s'|s, a, t) V_k(s') \right]$

## Implementation Considerations

1. **State Space Reduction**: The full state space can be quite large. Consider aggregating states or using approximation methods if needed.

2. **Uniform Continuous-Time Markov Decision Process (UCTMDP)**: Since both arrival and service processes are continuous-time, this problem can be formulated as a UCTMDP and solved using the equivalent discrete-time MDP with uniformization.

3. **Practical Queue Management**: The optimal policy may include considerations like:
   - When both queues have similar expected waiting times, route simple calls to Agent 2 and complex calls to Agent 1
   - When one queue is significantly shorter, route to that queue regardless of query type
