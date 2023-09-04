#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:48:39 2023

@author: sahil
"""

'''
Import Libraries
'''
import pandas as pd
import numpy as np
import random

np.random.seed(99163)

'''
Define Environment
'''

'''Action Space'''
start_quantity = 70
step = 2.5
end_quantity = 104 + step
action_space = np.arange(start_quantity, end_quantity, step)

del start_quantity, step, end_quantity

'''State Space'''
shocks = [290, 310]
state_space = {}

for shock in shocks:
    for c1_choice in action_space:
        for c2_choice in action_space:
            state = shock - (c1_choice + c2_choice)
            state_space[(shock, c1_choice, c2_choice)] = state

del c1_choice, c2_choice


'''Generate Q spaces for all competitors'''
competitors = ['Competitor 1', 'Competitor 2']
   

# Generate Q-matrix for each competitor
Q_matrix = {}

for competitor in competitors:
    Q_matrix[competitor] = {}
    for state in state_space:
        if competitor == 'Competitor 1':
            Q_matrix[competitor][tuple(state)] = (tuple(state)[0] - tuple(state)[1] - tuple(state)[2]) * tuple(state)[1]
        else:
            Q_matrix[competitor][tuple(state)] = (tuple(state)[0] - tuple(state)[1] - tuple(state)[2]) * tuple(state)[2]


'''Annealing Schedule'''
def exponential_decay(initial_value, beta, num_points):
    x = np.linspace(initial_value, num_points, num_points)
    decayed_values = np.exp(-beta * x)
    return decayed_values


'''Generate Random Shocks'''
num_iterations = 10
num_loops = 100000
random_shocks = np.random.choice(shocks, size = (num_loops, num_iterations), replace = True)

'''Generate Decay depending on Annealing Schedule'''
beta = 4e-5
decay = exponential_decay(1, beta, 100000)


'''Generate Final Dictionary'''
results = {'action_1':[],
           'action_2':[],
           'price':[]
           }


'''Final Loop'''
exploration_rate_threshold = np.random.uniform(0,1)
exploration_rate = 1

for loop in range(num_loops):
    
    state = random.choice(list(state_space.keys()))
    
    for iteration in range(num_iterations):
        #Realize Shock
        shock = random_shocks[loop][iteration]

        #Choose Actions Independently
        action_1 = get_action(exploration_rate, action_space, shocks, Q_matrix, 'Competitor 1')
        action_2 = get_action(exploration_rate, action_space, shocks, Q_matrix, 'Competitor 2')

        #Get rewards
        price = shock - (action_1 + action_2)
        reward_1 = get_reward(action_1, price)
        reward_2 = get_reward(action_2, price)
                    

        #Get Next state
        new_state = tuple({shock, action_1, action_2})         
        
        
        #Get the RHS
        td_target1 = target_reward(reward_1, 'Competitor 1', new_state)
        td_target2 = target_reward(reward_1, 'Competitor 2', new_state)

        
        #Update Qmatrix
        update_qmatrix(state, td_target1, 'Competitor 1', alpha = 0.15)
        update_qmatrix(state, td_target2, 'Competitor 2', alpha = 0.15)
        
        #Add action and price decisions in results
        new_values = [action_1, action_2, price]
        results = {key: results[key] + [value] for key, value in zip(results.keys(), new_values)}

    
    print(loop)
    if loop < 10:
        exploration_rate = decay[loop]
    else:
        exploration_rate = 0
            
    
    
    '''All Functions'''
    
def get_action(exploration_rate, action_space, shock_space, Q_matrix, competitor_name):
    exploration_rate_threshold = np.random.uniform(0,1)
    
    if exploration_rate_threshold < exploration_rate:
        action = random.choice(action_space)
    else:
        search_tuple = (shock,)
        matching_tuples = [key for key in Q_matrix[competitor_name] if key[:len(search_tuple)] == search_tuple]
        if matching_tuples:
            highest_value = max(Q_matrix[competitor_name][key] for key in matching_tuples)
            for key in matching_tuples:
                if Q_matrix[competitor_name][key] == highest_value:
                    action = key[1]
    return action
    

def get_reward(action, price):
    return action * price


def target_reward(reward, competitor_name, state, discount_factor = 0.95):
    price = shock - (action_1 + action_2)
    search_tuple = (shock,)
    
    matching_keys = [key for key, value in state_space.items() if key[:len(search_tuple)] == search_tuple and value == price]
    
    max_value = np.max([Q_matrix[competitor_name][key] for key in matching_keys if key in Q_matrix[competitor_name]])
    
    value = reward + discount_factor * max_value
    return value


def update_qmatrix(old_state, value, competitor_name, alpha = 0.15):
    Q_matrix[competitor_name][old_state] = alpha * value + (1 - alpha)* Q_matrix[competitor_name][old_state]
    


