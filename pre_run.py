#!/usr/bin/env python3

import numpy as np
import os 
import pandas as pd
import pickle

"""
TODO: agent config generation, LSTM price forecast for first 7 prices
"""
def agent_configs():
    P_water = abs(np.random.normal(4,0.3,1)[0])
    W_arr = np.random.choice(np.arange(6,10,0.25))
    P_air = abs(np.random.normal(1,0.1,1)[0])
    ground_area = abs(np.random.normal(72,5,1)[0])
    height = abs(np.random.normal(5,0.1,1)[0])
    curr_temp_air = np.random.normal(22,1,1)[0]
    slope_air = abs(np.random.normal(0.4,0.05,1)[0])
    slope_water = abs(np.random.normal(1,0.1,1)[0])
    desire_temp_air = abs(np.random.normal(22,1,1)[0])
    desire_temp_water = abs(np.random.normal(60,2,1)[0])
    alpha_water = abs(np.random.normal(1,0.1,1)[0])
    alpha_air = abs(np.random.normal(1.5,0.2,1)[0])
    alpha_cost = abs(np.random.normal(1,0.1,1)[0])
    baseload = abs(np.random.normal(0.3,0.05,1)[0])
    solar = np.random.choice([True, False, False]) # 1/3 chance of being prosumer

    agent_dict = {  'P_water' : P_water,
                    'W_arr' : W_arr,
                    'P_air' : P_air,
                    'ground_area' : ground_area,
                    'height' : height,
                    'curr_temp_air' : curr_temp_air,
                    'slope_air' : slope_air,
                    'slope_water' : slope_water,
                    'desire_temp_air' : desire_temp_air,
                    'desire_temp_water' : desire_temp_water,
                    'alpha_water' : alpha_water,
                    'alpha_air' : alpha_air,
                    'alpha_cost' : alpha_cost,
                    'baseload' : baseload,
                    'solar': solar,
                    'water_usage' : 65.11/5/1000,
                    'vol_water' : 0.1514164,
                    'tank_area' : 3.9381,
                    'all_opts' : pd.read_csv('binaries.txt',index_col = False, header = None, delimiter = ' '),
                    'curr_temp_water' : 60}

    return agent_dict

cwd = os.getcwd()
agent_config_dir = os.path.join(cwd,'agent_config_files')
if not os.path.exists(agent_config_dir):
    os.mkdir(agent_config_dir)
num_of_agents = 30

# based on LSTM price forecast
price_list =  [10.4] * 4 + [20.8] * 2


"""
 writes the initial configuration of agents
 does not take into account of curr_t
 curr_t must be supplied in main.py
"""
for agent_id in range(num_of_agents):
    agent_dict = agent_configs()
    agent_dict['price_list'] = price_list
    name_of_pickle = "agent_id_" + str(agent_id) + "_params.p"
    output_file = os.path.join(agent_config_dir, name_of_pickle)
    pickle_file = open( output_file, "wb")
    pickle.dump(agent_dict, pickle_file)
    pickle_file.close()
    

