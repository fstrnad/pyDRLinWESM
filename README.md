# pyDRLinWESM
A small package for using Deep Reinforcement Learning within World-Earth System Models to discover sustainable management strategies. Deep Reinforcement Learning cannot only play Atari Games and Go but [point to sustainable management pathways as well](https://www.pik-potsdam.de/news/in-short/articial-intelligence-applying-201adeep-reinforcement-learning2018-for-sustainable-development). This repository contains a python implementation developed in the context of the [COPAN](https://www.pik-potsdam.de/research/projects/activities/copan/copan-introduction)
Collaboration at [Potsdam Institute for Climate Impact research](https://www.pik-potsdam.de/).

Please acknowledge and cite the use of this software and its authors when results are used in publications or published elsewhere.

You can use the following reference [[1]](#1):


Felix M. Strnad, Wolfram Barfuss, Jonathan F. Donges and Jobst Heitzig,
*Deep reinforcement learning in World-Earth system models to discover sustainable management strategies*,
Chaos, 2019
DOI: [10.1063/1.5124673](http://aip.scitation.org/doi/10.1063/1.5124673)

This package requires for usage:
 - tensorflow=1.14 (tensorflow>=2.0 runs as well, but is significantly slower in running time)
 - keras>=2.3. (in case of tensorflow2 this package is not nessesary anymore) 
 - pyviability (find at https://github.com/timkittel/ays-model/)
 - gym 

## Description
Increasingly complex, non-linear World-Earth system models are used for describing the dynamics of the biophysical Earth system and the socio-economic and socio-cultural World of human societies and their interactions. Identifying pathways towards a sustainable future in these models is a challenging and widely investigated task in the field of climate research and broader Earth system science.  This problem is especially difficult when caring for both environmental limits and social foundations need to be taken into account.

In this work, we propose to combine recently developed deep reinforcement learning (DRL) algorithms, with classical analysis of trajectories in the World-Earth system as an approach to extend the field of Earth system analysis by a new method.

## Agent-Environment Interface

 <img src="./figures/Agent_Environment_Interface_DQN_Learner_Interpretation-1.png">

This work proposes a new approach for using DRL within World-Earth system models. It uses the basic information if at time `t`  a certain state `s_t` is within the sustainability boundaries. The underlying mathematical foundation is based on an Markov Decission Process (MDP) which is translated to a concrete agent-environment setup.

In this context, the concept of the agent is solely defined by its action set. In our interpretation, the action set can be regarded as a collection of possible measures the international community could use to influence the system's trajectory. We interpret the different system management options in the described environments as distinct actions of the action set.

In contrast to other applications of DRL where rewards are defined by the application (e.g. scores in a computer game), in our application the reward functions are not a system feature but a parameter of the learning algorithm. We are free in our choice of the reward function and are guided in this choice by how well the chosen reward function helps the learner to achieve the actual goal. Since our ultimate objective is not to maximize some objectively given reward function but to stay within the boundaries we chose the reward functions accordingly. Reward functions can be both continuously and discontinuously changing.


## First Steps
You can try out the framework via the jupyter-notebook in the example folder examples/Example_DRL_environment.ipynb. This notebook guides you through the nessesary functions and improvements the DRL agent can use. 
You can as well just the example code in examples/example_code_one_run.py via

```
export PYTHONPATH=. (if not set yet)
python examples/example_code_one_run.py 0
```

## Setup for a single run
After the experience replay memory is filled with experiences from an agent that acts randomly in the environment, the learning process runs as follows.
The agent is trained for a fixed number of episodes. A start position within the boundaries is randomly drawn from a uniform distribution of states around the current state. The number of iteration steps during one single learning episode is limited to a maximum of `T`. The end of one learning episode is determined either when `T` is reached or ended prematurely at time `t` either when a boundary is crossed or when approximate convergence to a fixed point is detected. In the latter case, the remaining future rewards are estimated with a discounted reward sum for the remaining time `T-t` of the reward `r_t`. In any case, after the end of a learning episode, the environment is reset to time `t=t_0` and a new start point `s_{t_0}` within the boundaries of the environment is randomly drawn. 

## Neural Network setup
The neural network is based on the following architecture. The input layer of the size equaling the dimension of the state space is followed by two fully-connected hidden layers, each one consisting of 256 units. The output layer is a fully connected linear layer that provides an output value for each possible action in the action set, representing the estimated Q-value of that action for the state given by the inputs. For minimizing the loss function, instead of simple stochastic gradient descent (SGD) the Adam optimizer is used due to its better performance than SGD in DRL applications. 


## Environments
This package is written in a way for using DRL in Earth system models, mathematically formalized in a Markov decision process. This package includes two prototypes, the AYS Environment and the c:GLOBAL environment (see below). Extensions based on this framework might be a tool to discover and analyze management pathways in global policy evaluation and will possibly lead towards a deeper understanding of the impact of global governance policies.
Environments contain as an internal variable the current state s of the environment. Based on this state, an environment suitable for our framework requires just two functions:
 - `next_state, reward, done = Environment.step(action)` The step function implements the dynamics of the environment. The dynamics can be both stochastic or deterministic 
 - `reward_function()` You are free in your choice of the reward function and thus you can let you guide for your choice by how well the chosen reward function helps the learner to achieve the actual goal. We implemented for example our reward functions in the following simple, action-independent way: 
   - Survival Reward: Provide a reward of 1 if the state `s_t` is the within the boundaries, else 0.
   - Boundary distance reward: Calculate the distance of the state `s_t` to the sustainability boundaries in units of distance of the current state of the Earth to the boundaries, i.e. initially the reward is 1. 
 - `action_set` In the interpretation of this framework, the action set A can be regarded as a collection of possible measures the international community could use to influence the system's trajectory. You might interpret the different system management options in the described environments as distinct actions of the action set. 

  
  
Here, we provide a brief description of the two prototype Environments we used as toy examples for our framework. 
 - **AYS Environment** [[2]](#2)
     This model is a low-complexity model in three dimensions studied and described in detail in [[2]]. It includes parts of climate change, welfare growth and energy transformation. The AYS environment is an interesting minimum-complexity toy model for sustainability science because one can represent both the climate change planetary boundary and a wellbeing social foundation boundary in it by studying whether atmospheric carbon stock may stay below some threshold, and the economic output does not drop below some minimum value at the same time.tory offers a method for using a DRL-agent that is able to act and learn in variable manageable environment models of the Earth system in order to discover management strategies for sustainable development. 

 - **copan:GLOBAL Environment** [[3]](#3)
     The model describes a conceptual seven-dimensional World-Earth model considering the co-evolution of human society and global climate on millennial timescales, having seven dynamic variables, as well as several additional for the agent non-observable auxiliary variables. The model is meant for a qualitative understanding of their complex interrelations rather than for quantitative predictions. 
This co-evolutionary approach accounts more explicitly for a global carbon cycle as well as for the dynamics of the human population, physical capital, and energy production to simulate the influence of humans on global climate in the Anthropocene.

However, feel free to implement your own environments and test the framework on your own.


## References
<a id="1">[1]</a> 
Felix M. Strnad, Wolfram Barfuss, Jonathan F. Donges and Jobst Heitzig, 2019,
Deep reinforcement learning in World-Earth system models to discover sustainable management strategies,
Chaos,
DOI: [10.1063/1.5124673](http://aip.scitation.org/doi/10.1063/1.5124673)

<a id="2">[2]</a>
Tim Kittel, Rebekka Koch, Jobst Heitzig, Guillaume Deffuant, Jean-Denis Mathias  and Jürgen Kurths, 2017,
Operationalization of Topology of Sustainable Management to Estimate Qualitatively Different Regions in State Space,
arXiv: [1706.04542](https://arxiv.org/abs/1706.04542)


<a id="3">[3]</a>
Jan Nitzbon, Jobst Heitzig, and Ulrich Parlitz, 2017,
Sustainability, collapse and oscillations in a simple World-Earth model, 
Environmental Research Letters 12.7,
DOI: [10.1088/1748-9326/aa7581](https://iopscience.iop.org/article/10.1088/1748-9326/aa7581)

## License
pyDRLinWESM is [BSD-licensed (2 clause)](./LICENSE)
