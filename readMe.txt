python = 3.7
packages = tensorflow

Agent_Critic: the decision makers and critics are defined in this script, It's an implementation of the MADDPG paper.

Networrk: All the elements necessary for the environment(the Network class) in which the agents will navigate and learn. The evironment is made out of several nodes and 
traffic requests between these nodes. Several agents can play the game, each round every agent can build one connection between two nodes, the connections are 
oriented. By creating connections agents can meet the traffic demands, and if it the case traffic will flow through their connections. The bigger the flow is in an agent's 
connection the bigger its reward. Rewards are computed between each round. The requests do not change over time, i.e. they remain constant.

Algorithm: In this script the game takes place, the agents and the environment interact and the agents learn from the returns of the environment.