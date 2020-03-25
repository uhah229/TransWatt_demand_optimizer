#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import random
from demandoptimizer import DemandOptimizer
from tqdm import tqdm
import pickle
from pre_run import generate_agent_cfg, clean_up, agent_configs
from forecaster import load_model, get_price_from_Q, make_n_predictions, create_predictions
from aggregator import aggDemandFiles, findIntersect, get_clearing_prices


# cleanup and generate agent configuration files
cwd = os.getcwd()
clean_up(cwd)
generate_agent_cfg(num_of_agents=30)

# generate first 6 prices 
# mcp = np.array([10.1] * 4 + [20.8] * 16 + [14.4] * 24 + [20.8] * 8 + [10.1] *
#         44 )
system_historical_demand_file = "synthesized.csv"
model = load_model(os.path.join(cwd,"energy_prediction_model"))
df = pd.read_csv(os.path.join(cwd,system_historical_demand_file), header=None)
time = np.arange(0,len(df)*15,15)
Q = pd.DataFrame(dict(energy=df.values.flatten()),index=time,columns=['energy'])
predicted_Q = make_n_predictions(Q,model,n=6,time_steps=24).values
predicted_prices = [get_price_from_Q(Q) for Q in predicted_Q]


path_to_configs = os.path.join(cwd,'agent_config_files')
list_of_configs = os.listdir(path_to_configs)


for t in np.arange(6,30*30,0.25):
    curr_t = t % 24
    next_timestep_system_demand = 0
    for config_fname in list_of_configs:
        agent_id = int(config_fname.split("_")[2])
        full_config_path = os.path.join(path_to_configs,config_fname)

        # load agent pickle
        with open(full_config_path,'rb') as fp:
            agent_dict = pickle.load(fp)

        agent_dict['price_list'] = predicted_prices
        agent_dict['curr_t'] = curr_t
        dem_opt = DemandOptimizer(agent_dict,agent_id)

        # update the air/water temp into the pickle file
        updated_air_temp, updated_water_temp, next_timestep_demand = dem_opt.get_response()
        next_timestep_system_demand += next_timestep_demand
        agent_dict['curr_temp_air'] = updated_air_temp
        agent_dict['curr_temp_water'] = updated_water_temp
        with open(full_config_path, 'wb') as fp:
            pickle.dump(agent_dict,fp)

    with open(system_historical_demand_file,'a+') as fp:
        fp.write(str(next_timestep_system_demand[0])+"\n")

    Q = Q.append({'energy' : next_timestep_system_demand[0]}, ignore_index=True)
    predicted_Q = make_n_predictions(Q,model,n=6,time_steps=24).values
    predicted_prices = [get_price_from_Q(Q) for Q in predicted_Q]
    print("Predicted prices:",predicted_prices)

    market_clearing_price = get_clearing_prices()
    predicted_prices[0:2] = market_clearing_price[1:3]
    print("Predicted Q", predicted_Q)
    print("Clearing prices: ",market_clearing_price)
    print("Finished timestep:",t)