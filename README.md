# pyDRLinWESM
A small package for using Deep Reinforcement Learning with World-Earth System Models.

Please acknowledge and cite the use of this software and its authors when results are used in publications or published elsewhere.

You can use the following reference:


Felix M. Strnad, Wolfram Barfuss, Jonathan F. Donges and Jobst Heitzig,
Deep reinforcement learning in World-Earth system models to discover sustainable management strategies,
Chaos, 2019

This package requires for usage:
 - tensorflow>=2.0
 - pyviability (find at https://github.com/timkittel/ays-model/)

## Description
Increasingly complex, non-linear World-Earth system models are used for describing the dynamics of the biophysical Earth system and the socio-economic and socio-cultural World of human societies and their interactions. Identifying pathways towards a sustainable future in these models is a challenging and widely investigated task in the field of climate research and broader Earth system science.  This problem is especially difficult when caring for both environmental limits and social foundations need to be taken into account.

In this work, we propose to combine recently developed deep reinforcement learning (DRL), with classical analysis of trajectories in the World-Earth system as an approach to extend the field of Earth system analysis by a new method.
Based on the concept of the agent-environment interface, we develop a method for using a DRL-agent that is able to act and learn in variable manageable environment models of the Earth system in order to discover management strategies for sustainable development.

## Setup for a single run
After the experience replay memory is filled with experiences from an agent that acts randomly in the environment, the learning process runs as follows.
The agent is trained for a fixed number of episodes. A start position within the boundaries is randomly drawn from a uniform distribution of states around the current state. The number of iteration steps during one single learning episode is limited to a maximum of $ T $. The end of one learning episode is determined either when $ T $ is reached or ended prematurely at time $ t $ either when a boundary is crossed or when approximate convergence to a fixed point is detected. In the latter case, the remaining future rewards are estimated with a discounted reward sum for the remaining time $ T-t $ of the reward $ r_t $. In any case, after the end of a learning episode, the environment is reset to time $ t=t_0 $ and a new start point $ s_{t_0} $ within the boundaries of the environment is randomly drawn. 

## Neural Network setup
The neural network is based on the following architecture. The input layer of the size equaling the dimension of the state space is followed by two fully-connected hidden layers, each one consisting of 256 units. The output layer is a fully connected linear layer that provides an output value for each possible action in the action set, representing the estimated Q-value of that action for the state given by the inputs. For minimizing the loss function, instead of simple stochastic gradient descent (SGD) the Adam optimizer is used due to its better performance than SGD in DRL applications.
