# Multi Agent Collaboration with Multi Agent Deep Deterministic Policy Gradient

## Getting started

1. clone and set up environment by running ```git clone https://github.com/thanakijwanavit/deep_rl_env.git```
2. install docker according to the instruction [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
3. initialize environment by running ```bash ./deep_rl_env/start_script.sh``` --> this will create a docker container which can be accessed through ```bash ./deep_rl_env/monitor_script.sh```
4. within the docker container, run ```git clone git@github.com:thanakijwanavit/tennis_maddpg_collaboration_competition.git```
5. download the necessary environment from this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
6. extract zip file ```unzip Tennis_Linux.zip```
7. run ```python train.py``` to train the model and save the artifacts


## Goal

Training a two agent system to keep tennis ball in play in an Environment similar to [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) by Udacity.

The goal is to keep the ball in play for as long as possible

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5


## Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.
The environment is considered solved when the average (over 100 episodes) of those scores is at least +0.5.


## Learning Algorithm

1. Baseline 

A random choice agent was established as a baseline in order to compare the trained model agains. This gives a result of 0-0.02 scores over 100 episodes


2. Environment

### Agents

there are 2 agents in this environment

### Action spaces

Navigation project was limited to four discrete actions: left, right, forward, backward.

### MADDPG Agent
The ddpg model consists of 2 sets of DDPG network, each consists of a critic and an agent, each has 2 hidden layer of 256 and 128.
the original [DDPG model](https://arxiv.org/pdf/1509.02971.pdf) is extended to cover multiple agents.


![](https://github.com/thanakijwanavit/DeepRL-P3-Collaboration-Competition/raw/7ff1d561c315b6e31e02aa40e848b8d3d5f9cbf0/assets/multi-agent-actor-critic.png)



### Actor-Critic Method

Actor-critic methods leverage the strengths of both policy-based and value-based methods. Actor choose an action to perform while critic evaluate the action giving it a score.
In the file ```maddpg_agent.py```, the agent is defined.

In the file ```train.py```, the agent is initialized with the code below.

```python
# Actor Network (w/ Target Network)
self.actor_local = Actor(state_size, action_size, random_seed).to(device)
self.actor_target = Actor(state_size, action_size, random_seed).to(device)
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

# Critic Network (w/ Target Network)
self.critic_local = Critic(state_size, action_size, random_seed).to(device)
self.critic_target = Critic(state_size, action_size, random_seed).to(device)
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

### Hyperparameter

#### Noise Parameters
```python
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15          # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
```

#### Learning Interval
```python
LEARN_EVERY = 1         # learning interval (no. of episodes)
LEARN_NUM = 5           # number of passes per learning step
```


## Training

the environment wast solved in 1422 episodes as shown



![](http://file.hatari.cc/PkeEJ/training_log.jpg)
![](http://file.hatari.cc/FcQ3n/training_plot27_19:36.png)

## Future Improvements

* Prioritized experience replay
* further Hyperparameter optimization
* batch normalization
* Further stability
the number of episodes required to solve the network is inconsistent across multiple runs. Perhaps some hyperparameters are causing the instability
* LSTM instead of neural network
This may lead to a better agent which can make use of its memory
