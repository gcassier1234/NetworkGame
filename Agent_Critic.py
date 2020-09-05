# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:02:22 2020

@author: gcass
"""

import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import logging
import time
import os
import random

class Actor():
    def __init__(self, obssize, actsize, clip_eps, step_per_train, sess, optimizer, global_step, connections):
        """
        obssize: size of the states
        actsize: size of the actions
        """
        # Building the prediction graph
        L = 100
        M = 50
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L],stddev=0.01))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.01))
        W2 = tf.Variable(tf.truncated_normal([L, M],stddev=0.01))
        B2 = tf.Variable(tf.truncated_normal([M], stddev=0.01))
        W3 = tf.Variable(tf.truncated_normal([M, actsize],stddev=0.01))
        B3 = tf.Variable(tf.truncated_normal([actsize], stddev=0.01))
        
        
        Z1 = tf.sigmoid(tf.matmul(state, W1) + B1)
        Z2 = tf.sigmoid(tf.matmul(Z1, W2) + B2)
        logit = tf.matmul(Z2, W3) + B3
        #logit = tf.matmul(Z1, W2) + B2
        
        # An array of legal action representation (legal=1, illegal=0)
        legal_actions = tf.placeholder(tf.float32, shape=[None,actsize])

        # Manually calculating softmax over selected (legal) indices only
        exp_logit = tf.exp(logit)
        prob = tf.multiply(tf.divide(exp_logit, tf.reduce_sum(tf.multiply(exp_logit, legal_actions))),legal_actions)
        #prob = tf.nn.softmax(tf.multiply(logit, legal_actions))  # prob is of shape [None, actsize]
        
        # BUILD LOSS: Proximal Policy Optimization
        # Advantage estimation from the experiences
        Q_estimate = tf.placeholder(tf.float32, [None])
        # Action probabilities that the previous iteration of network would predict
        old_prob = tf.placeholder(tf.float32, [None, actsize])
        # Action indices that were picked in the experience
        actions = tf.placeholder(tf.int32, [None])
        # Encode those actions into one_hot encoding of length actsize
        actions_one_hot = tf.one_hot(actions, depth=actsize)
        # Select only the relevant probability for new and old probabilities
        prob_i = tf.reduce_sum(tf.multiply(prob, actions_one_hot), axis=1)
        old_prob_i = tf.reduce_sum(tf.multiply(old_prob, actions_one_hot), axis=1)
        
        ratio = tf.divide(prob_i, old_prob_i)

        surrogate_loss = tf.negative(tf.reduce_mean(tf.minimum(
            tf.multiply(ratio, Q_estimate),
            tf.multiply(tf.clip_by_value(ratio, 1-clip_eps, 1+clip_eps), Q_estimate)
        ))) + tf.reduce_sum(prob*tf.log(prob + 1e-9))
        
        
        self.train_op = optimizer.minimize(surrogate_loss, global_step = global_step)
        
        # some bookkeeping
        self.state          = state
        self.prob           = prob
        self.old_prob       = old_prob
        self.actions        = actions
        self.legal_actions  = legal_actions
        self.Q_estimate     = Q_estimate
        self.loss           = surrogate_loss
        self.clip_eps       = clip_eps
        self.step_per_train = step_per_train
        self.optimizer      = optimizer
        self.sess           = sess
        
        self.Connections    = connections #represents the connection between nodes the agent already owns
        
    def compute_prob(self, states, legal_actions):
        """
        compute prob over actions given states pi(a|s)
        states: numpy array of size [numsamples, obssize]
        legal_actions: array of size [numsamples, actsize], 0 if illegal 1 if legal in that state
        return: numpy array of size [numsamples, actsize]
        """
        
        return self.sess.run(self.prob, feed_dict={self.state:states, self.legal_actions:legal_actions})

    def train(self, states, actions, Qs, old_prob, legal_actions):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        for i in range(self.step_per_train):
            self.sess.run(self.train_op, feed_dict={self.state:states, self.actions:actions, self.Q_estimate:Qs, self.old_prob:old_prob, self.legal_actions:legal_actions})
            
class Critic():
    def __init__(self, obssize, sess, optimizer, global_step):
        # YOUR CODE HERE
        # need to implement both prediction and loss
        L = 50
        #M = 20
        state = tf.placeholder(tf.float32, [None, obssize])
        W1 = tf.Variable(tf.truncated_normal([obssize, L], stddev=0.01))
        B1 = tf.Variable(tf.truncated_normal([L], stddev=0.01))
        W2 = tf.Variable(tf.truncated_normal([L, 1],stddev=0.01))
        B2 = tf.Variable(tf.truncated_normal([1], stddev=0.01))
        #W3 = tf.Variable(tf.truncated_normal([M, 1],stddev=0.1))
        #B3 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
        
        Z1 = tf.sigmoid(tf.matmul(state, W1) + B1)
        #Z2 = tf.sigmoid(tf.matmul(Z1, W2) + B2)
        val = tf.matmul(Z1, W2) + B2
        
        target = tf.placeholder(tf.float32, [None, 1])
        loss = tf.losses.mean_squared_error(target, val)
        self.train_op = optimizer.minimize(loss, global_step = global_step)
        
        self.state = state
        self.val = val
        self.target = target
        self.loss = loss
        self.sess = sess
        self.optimizer = optimizer

    def compute_values(self, states):
        """
        compute value function for given states
        states: numpy array of size [numsamples, obssize]
        return: numpy array of size [numsamples]
        """
        # YOUR CODE HERE
        return self.sess.run(self.val, feed_dict={self.state:states})

    def train(self, states, targets):
        """
        states: numpy array
        targets: numpy array
        """
        # YOUR CODE HERE
        return self.sess.run(self.train_op, feed_dict={self.state:states, self.target:targets})


def discounted_rewards(r, lmbda):
    """ take 1D float array of rewards and compute discounted bellman errors """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * lmbda + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)