#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import random
from demandoptimizer import DemandOptimizer
from tqdm import tqdm
import pickle

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
    alpha_water = abs(np.random.normal(10,0.1,1)[0])
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


# mcp_file = 'blabla.csv' # Market Clearing Prices file
# mcp = np.loadtxt(mcp_file,delimiter=',')
# mcp = [13.4,0.3,9.4,6.5,9.4,13.4]
mcp = np.array([10.1] * 4 + [20.8] * 16 + [14.4] * 24 + [20.8] * 8 + [10.1] *
        44 )
# mcp = [13.4,13.4,13.4,9.4,9.4,9.4]
# agent_dict = agent_configs()
# agent_dict['price_list'] = mcp
# agent_dict['curr_t'] = 5
# dem_opt = DemandOptimizer(agent_dict)
# print(dem_opt.get_response())

# num_agents = 30
# for agent_id in range(num_agents):
#     agent_dict = agent_configs()
#     agent_dict['price_list'] = mcp
#     agent_dict['curr_t'] = 12
#     dem_opt = DemandOptimizer(agent_dict,agent_id)
#     dem_opt.get_response()

cwd = os.getcwd()
path_to_configs = os.path.join(cwd,'agent_config_files')
list_of_configs = os.listdir(path_to_configs)

"""TODO
Update temperatures in DemandOptimizer DONE
Update prices in main
    - need aggregator
    - LSTM
"""

for t in tqdm(np.arange(6,30,0.25)):
    curr_t = t % 24
    for config_fname in list_of_configs:
        agent_id = int(config_fname.split("_")[2])
        full_config_path = os.path.join(path_to_configs,config_fname)

        # load agent pickle
        with open(full_config_path,'rb') as fp:
            agent_dict = pickle.load(fp)

        indices = ((np.arange(curr_t,curr_t+1.5,0.25)*4-24) % 96).astype("int")
        agent_dict['price_list'] = mcp[indices]
        agent_dict['curr_t'] = curr_t
        dem_opt = DemandOptimizer(agent_dict,agent_id)

        # update the air/water temp into the pickle file
        updated_air_temp, updated_water_temp = dem_opt.get_response()
        agent_dict['curr_temp_air'] = updated_air_temp
        agent_dict['curr_temp_water'] = updated_water_temp
        with open(full_config_path, 'wb') as fp:
            pickle.dump(agent_dict,fp)
