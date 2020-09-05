# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:06:37 2020

@author: gcass
"""

from Network import *
from Agent_Critic import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import logging
import time
import os
import random
"""
eval_func allows evaluation of the actors at a given point of the training
"""
def eval_func(actors, actsize, nOperators, network):
    eval_episodes = 15
    
    avgRewards = [] #avg reward for each elevator in each trajectory
    
    for ite in range(eval_episodes):
        
        network.reset()
        rsum = [0 for operator in range(nOperators)]
        
        for game in range (nGames) :
            
            network.reset()
            states           = network.Connections.flatten()
            rewards          = network.rewardAll()
            actions = []
            
            for idOperator, agent in enumerate(actors) :
                reverse_legal_actions   = network.Connections[idOperator].flatten()
                legal_actions            = np.where(((reverse_legal_actions==0)|(reverse_legal_actions==1)), reverse_legal_actions^1, reverse_legal_actions)
                boolean_legal_actions   =  np.append(legal_actions,1)
                prob   = agent.compute_prob(np.expand_dims(state,0), np.expand_dims(boolean_legal_actions, 0)).flatten()
                action = np.random.choice(np.arange(actsize), p=prob)
                actions.append(action)
                b_legal_actions.append(boolean_legal_actions)
                
            # record
            for operator in range(nOperators):
                
                rsum[operator] += rewards[operator]
            
            for operator in range(nOperators) :

                if actions[idOperator] != nNodes**2 : 
                    
                    tailNode, headNode = np.unravel_index(actions[idOperator], (nNodes,nNodes))
                    network.addConnection(idOperator, tailNode, headNode)

        avgRewards.append([rsum[operator ] for operator in range(nOperators)])

    avgRewards = np.array(avgRewards)
    print("{:40s}: {}, Average:{}".format("Average reward of elevators in each episode", np.round(np.mean(avgRewards, axis=1)), np.mean(avgRewards)))
    


"""
############################################################################# Problem settings #############################################################################
"""

# NeuralNetwork parameters (NeuralNets = operators and critics)
#starter_learning_rate      = 3e-4
lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
lr_beta                    = 3e-5 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
nGames                     = 10    # Number of games for each iteration 
nMoves                     = 5    # Number of moves played by each player
iterations                 = 100   # total num of iterations
gamma                      = 0.99 # discount
lmbda                      = 0.5  # GAE estimation factor
clip_eps                   = 0.2
step_per_train             = 3

# Environment parameters
nOperators         = 2
nNodes             = 5


# Initialize environment
links              = {}
nodes              = {i:Node(i) for i in range(nNodes)}
demands            = {(0,1):Demand(0,1,5), (0,2):Demand(0,2,5)} #The index of a demande must be equal to the couple (departure, destination)
network            = Network(nodes, links, demands, nOperators)
obssize            = nOperators*nNodes**2 #The obsarvation space is made out of the flattened connection matrixe of the operators
actsize            = nNodes**2 + 1 #The last possible action corresponds to inactivity, no connection is built by the operator during this round


## Initialize Tensorflow/network stuff
sess               = tf.Session()
actors             = []
critics            = [] 

for i in range(nOperators):
    
    global_step_alpha          = tf.Variable(0, trainable=False)
    global_step_beta           = tf.Variable(0, trainable=False)
    optimizer_p                = tf.train.AdamOptimizer(lr_alpha)
    optimizer_v                = tf.train.AdamOptimizer(lr_beta)
    actors.append(Actor(obssize, actsize, clip_eps, step_per_train, sess, optimizer_p, global_step_alpha, np.zeros((nNodes,nNodes))))
    critics.append(Critic(obssize, sess, optimizer_v, global_step_beta)) 
    
# Load saved model
CHECKPOINT_DIR = "./ckpts2/" # specify path to the checkpoint file
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Load a previous checkpoint if it exists
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
"""
if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
"""

"""
############################################################################# Iteration of the algorithm #############################################################################
"""
for ite in range(0,iterations):   
    
    OBS        = []                                         # observations, they are the same for every operators, thus there is no need for a list of obs for each operator
    ACTS       = {agent:[] for agent in range(nOperators)}  # actions
    ADS        = {agent:[] for agent in range(nOperators)}  # advantages for actors
    TARGETS    = {agent:[] for agent in range(nOperators)}  # targets for critics
    LEGAL_ACTS = {agent:[] for agent in range(nOperators)}  # legal actions encoded in 0's and 1's

    for game in range (nGames) :
        
        network.reset()                                  # network resets before a new game
        
        obss = []                                        # the observations are the same among all the operators 
        acts = [[] for agent in range(nOperators)]       # actions
        rews = [[] for agent in range(nOperators)]       # instant rewards for one trajectory
        legal_acts = [[] for agent in range(nOperators)] # legal actions encoded in 0's and 1's

        for move in range(nMoves) :

            state            = network.Connections.flatten()   #The state is made out of the flattened connection matrixe of the operators
            rewards          = network.rewardAll()             #rewards are computed for all operators
            actions          = [] # stores the action of every operator, one action per operator
            b_legal_actions  = [] # Boolean legal actions for one decision epoch! from all agents, one array representing leagal actions for each operator
            
            
            
            for idOperator, agent in enumerate(actors) :
                """
                Obtain legal actions and encoding them into 1's and 0's
                the legal action array is the flattened version of the reversed connection matrix of the operator
                (whas is meant by reversed is that 1 and 0 are inverted)
                This means that an operator can't creat a connection where it has already created one.
                Because we don't want an operator to create a connection between two same nodes
                ,for each operator the diaglonal elements of the connection matrix are preseted to 1, 
                """
                reverse_legal_actions   = network.Connections[idOperator].flatten()
                legal_actions            = np.where(((reverse_legal_actions==0)|(reverse_legal_actions==1)), reverse_legal_actions^1, reverse_legal_actions)
                boolean_legal_actions   = np.append(legal_actions,1)

                # Probability over all actions (but illegal ones will have probability of zero)
                prob   = agent.compute_prob(np.expand_dims(state,0), np.expand_dims(boolean_legal_actions, 0)).flatten()
                action = np.random.choice(np.arange(actsize), p=prob)
                actions.append(action)
                b_legal_actions.append(boolean_legal_actions)

            # record
            obss.append(state)
            for idOperator in range(nOperators) :
                
                acts[idOperator].append(actions[idOperator])
                rews[idOperator].append(rewards[idOperator]) # Need to add negation reward if the environment returns cost not reward
                legal_acts[idOperator].append(b_legal_actions[idOperator])
            
            
            """
            the decisions made by the operators are injected in the environment
            """
            for idOperator in range(nOperators) :
                """
                # if the action taken by the operator is equals to the last element of the action array (action array = [0, 1, 2, ..., nNodes*nNodes], see line 181) 
                it means the operator will do nothing during the round
                """
                if actions[idOperator] != nNodes**2 : 
                    tailNode, headNode = np.unravel_index(actions[idOperator], (nNodes,nNodes))
                    network.addConnection(idOperator, tailNode, headNode)
            
            """
            if game == 0 :
                
                print("####first agent action####")
                print (actions[0])
                print('----first agent connection matrix----')
                print(network.Connections[0])
                print("####second agent action####")
                print (actions[1])
                print('----second agent connection matrix----')
                print(network.Connections[1])
                print('######################################')
            """


        # Discard the first reward and the last actions
        # because each round the reward we observe correspond to the previous episode
        # We keep the last state because it serves as the next_state to the second to last state
        for idOperator  in range(nOperators) :
            acts[idOperator ]       = acts[idOperator ][:-1]
            legal_acts[idOperator ] = legal_acts[idOperator ][:-1]
            rews[idOperator ]       = rews[idOperator ][1: ]
            
        # Compute discount sum of rewards from instant rewards
        returns         = [discounted_rewards(rews[idOperator], gamma) for idOperator in range(nOperators)]
        
        #print("returns of one traject:", returns)
        # Compute the GAE
        vals            = np.array([critics[idOperator].compute_values(obss) for idOperator in range(nOperators)])
        bellman_errors  = [np.array(rews[idOperator]) + (gamma*vals[idOperator,1:] - vals[idOperator,:-1]).reshape(-1) for  idOperator in range(nOperators)]
        GAE             = [discounted_rewards(bellman_errors[idOperator], gamma*lmbda)  for  idOperator in range(nOperators)]
        # The last val is the value estimation of last state, which we discard
        #GAE             = {agent: list(np.array(returns[agent]) - v.flatten()[:-1]) for agent,v in vals.items()}
        #print("Advantage estimation:", GAE)
        
        # Record for batch update
        OBS += obss[:-1]
        for idOperator in range(nOperators) :
            TARGETS[idOperator] += returns[idOperator]
            ACTS[idOperator]    += acts[idOperator]
            ADS[idOperator]     += GAE[idOperator]
            LEGAL_ACTS[idOperator] += legal_acts[idOperator]
    
    #print('training is starting')
    OBS = np.array(OBS)
    for agent in range(nOperators) :
        # update baseline
        TARGETS[agent]  = np.array(TARGETS[agent])
        ACTS[agent]     = np.array(ACTS[agent])
        ADS[agent]      = np.array(ADS[agent])
        critics[agent].train(OBS, np.reshape(TARGETS[agent], [-1,1]))
    
        # update policy
        legal_actions = np.array(LEGAL_ACTS[agent])
        old_prob      = actors[agent].compute_prob(OBS, legal_actions)
        actors[agent].train(OBS, ACTS[agent], ADS[agent], old_prob, legal_actions)  # update
    
    """
    every ten iteration the actors and critics are saved
    """
    if (ite%10 == 0) and (ite != 0) :
        saver.save(sess, CHECKPOINT_DIR+"model_{}.ckpt".format(ite))
        print("MODEL SAVED AT ITERATION {}".format(ite))
    
    
    """
    if ite%20 == 0:
        eval_func(actors, actsize, nOperators, network)
    """