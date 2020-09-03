# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:06:37 2020

@author: gcass
"""

from Network import *
from Agent_Critic import *
import tensorflow as tf
import numpy as np
import logging
import time
import os
import random


logging.disable(logging.DEBUG)

# NeuralNetwork parameters
#starter_learning_rate      = 3e-4
lr_alpha                   = 3e-4 # tf.train.exponential_decay(starter_learning_rate, global_step_alpha, 100, 0.95, staircase=True)
lr_beta                    = 3e-5 # tf.train.exponential_decay(starter_learning_rate, global_step_beta, 100, 0.95, staircase=True)
nGames                     = 32   # Number of games for each iteration 
nMoves                     = 2    # Number of moves played by each player
iterations                 = 1000 # total num of iterations
gamma                      = 0.99 # discount
lmbda                      = 0.5    # GAE estimation factor
clip_eps                   = 0.2
step_per_train             = 3

# Environment parameters
nOperators         = 2
nNodes             = 3 


# Initialize environment
links              = {}
nodes              = {i:Node(i) for i in range(nNodes)}
demandes           = {(1,3):Demand(1,3,100)}
network            = Network(nodes, links, demands)
obssize            = len(network.Nodes)**2
actsize            = len(network.Nodes)**2
logger.warning("Action size: {:5d}, State size: {:5d}".format(actsize, obssize))
# TODO: God's eye obssize

## Initialize Tensorflow/network stuff
sess               = tf.Session()
actors             = []
critics            = []
for i in range(nOperator):
    global_step_alpha          = tf.Variable(0, trainable=False)
    global_step_beta           = tf.Variable(0, trainable=False)
    optimizer_p                = tf.train.AdamOptimizer(lr_alpha)
    optimizer_v                = tf.train.AdamOptimizer(lr_beta)
    actors.append(Actor(obssize, actsize, clip_eps, step_per_train, sess, optimizer_p, global_step_alpha, np.zeros((nNodes,Nodes))))
    critics.append(Critic(obssize, sess, optimizer_v, global_step_beta)) 
    
# Load saved model
CHECKPOINT_DIR = "./ckpts/" # specify path to the checkpoint file
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Load a previous checkpoint if it exists
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
    
# main iteration
    for ite in range(iterations):    

        if ite%10 == 0:
            saver.save(sess, CHECKPOINT_DIR+"model_{}.ckpt".format(ite))
            print("MODEL SAVED AT ITERATION {}".format(ite))
        
        OBS        = {agent:[] for agent in range(nOperators)}  # observations
        ACTS       = {agent:[] for agent in range(nOperators)}  # actions
        ADS        = {agent:[] for agent in range(nOperators)}  # advantages for actors
        TARGETS    = {agent:[] for agent in range(nOperators)}  # targets for critics
        LEGAL_ACTS = {agent:[] for agent in range(nOperators)}  # legal actions encoded in 0's and 1's

        for games in range(nGames):
            # record for each episode
            obss = {agent:[] for agent in range(nOperators)}  # observations
            acts = {agent:[] for agent in range(nOperators)}  # actions
            rews = {agent:[] for agent in range(nOperators)}  # instant rewards for one trajectory
            legal_acts = {agent:[] for agent in range(nOperators)} # legal actions encoded in 0's and 1's

            obs = network.Connections 
            

            epi_step = 0
            # Randomly step through the environment for some steps before officially starting to train
            random_start = np.random.randint(50,1000)
            seen_psngr = False
            for move in range(nMoves):
                epi_step += 1
                decision_agents  = obs["decision agents"]
                states           = obs
                rewards          = obs["rewards"]
                actions          = []
                b_legal_actions  = [] # Boolean legal actions for one decision epoch! from all agents
                # take actions
                if epi_step <= random_start:
                    actions = [random.sample(env.legal_actions(agent), 1)[0] for agent in decision_agents]
                    newobs = env.step(actions)
                    obs = newobs
                    continue
                #if epi_step == random_start+1:
                #    print("randomly started at step", epi_step)
                #    env.render()
                env.spawnRates = np.zeros(env.nFloor)
                
                for idx, agent in enumerate(actors):
                    # Obtain legal actinos and encode them into 1's and 0's
                    legal_actions                        = agent.Connections.flatten()
                    boolean_legal_actions                = legal_actions
    
                    # Probability over all actions (but illegal ones will have probability of zero)
                    prob   = actors[agent].compute_prob(np.expand_dims(states[idx],0), np.expand_dims(boolean_legal_actions, 0)).flatten()
                    action = np.random.choice(np.arange(actsize), p=prob)
                    actions.append(action)
                    b_legal_actions.append(boolean_legal_actions)

                # record
                for idx, agent in enumerate(decision_agents):
                    obss[agent].append(states[idx])
                    acts[agent].append(actions[idx])
                    rews[agent].append(rewards[idx]) # Need to add negation reward if the environment returns cost not reward
                    legal_acts[agent].append(b_legal_actions[idx])
                
                # update
                if env.no_passenger():
                    if seen_psngr:
                        #print("Took {} to reach empty state".format(epi_step-random_start))
                        #env.render()
                        break
                else:
                    seen_psngr = True

                newobs = env.step(actions)
                obs = newobs

            # logger.warning("Episode lasted: {:5d} step!".format(epi_step))

            # Discard the first reward and the last actions
            # because each round the reward we observe correspond to the previous episode
            # We keep the last state because it serves as the next_state to the second to last state
            for agent in range(nOperators):
                acts[agent] = acts[agent][:-1]
                legal_acts[agent] = legal_acts[agent][:-1]
                rews[agent] = rews[agent][1: ]
                
            # Compute discount sum of rewards from instant rewards
            returns         = {agent: discounted_rewards(rews[agent], gamma) for agent in rews}
            #print("returns of one traject:", returns)
            # Compute the GAE
            vals            = {agent: critics[agent].compute_values(ob) for agent, ob in obss.items()}
            bellman_errors  = {agent: np.array(rews[agent]) + (gamma*val[1:] - val[:-1]).reshape(-1) for agent, val in vals.items()}
            GAE             = {agent: discounted_rewards(errors, gamma*lmbda) for agent, errors in bellman_errors.items()}
            # The last val is the value estimation of last state, which we discard
            #GAE             = {agent: list(np.array(returns[agent]) - v.flatten()[:-1]) for agent,v in vals.items()}
            #print("Advantage estimation:", GAE)
            
            # Record for batch update
            for agent in range(nOperators):
                TARGETS[agent] += returns[agent]
                OBS[agent]     += obss[agent][:-1]
                ACTS[agent]    += acts[agent]
                ADS[agent]     += GAE[agent]
                LEGAL_ACTS[agent] += legal_acts[agent]
        
        for agent in range(nOperators):
            # update baseline
            TARGETS[agent]  = np.array(TARGETS[agent])
            OBS[agent]      = np.array(OBS[agent])
            ACTS[agent]     = np.array(ACTS[agent])
            ADS[agent]      = np.array(ADS[agent])
            critics[agent].train(OBS[agent], np.reshape(TARGETS[agent], [-1,1]))
        
            # update policy
            legal_actions = np.array(LEGAL_ACTS[agent])
            old_prob      = actors[agent].compute_prob(OBS[agent], legal_actions)
            actors[agent].train(OBS[agent], ACTS[agent], ADS[agent], old_prob, legal_actions)  # update

        if ite%20 == 0:
            eval_func(actors, actsize, nOperators, env)