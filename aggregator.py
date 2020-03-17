import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from glob import glob

def aggDemandFiles(files_dir = 'agents_demand_files/', num_prices = 4, num_periods = 6):
    demand_files_list = glob(files_dir+'*demand.csv')
    num_agents = len(demand_files_list)
    demand_matrix = np.zeros((num_prices,num_agents,num_periods))
    for agent in range(num_agents):
        demand_file = demand_files_list[agent]
        demand_matrix[:,agent,:] = pd.read_csv(demand_file,header=None)
    return demand_matrix

def findIntersect(demand_curve,supply_curve,prices):
    diff_curve = supply_curve - demand_curve
    where_positive = np.where(diff_curve > 0)
    candidate_price_index = where_positive[0][0]
    candidate_price = prices[candidate_price_index]
    if (demand_curve[candidate_price_index] <= supply_curve[candidate_price_index-1]) and (candidate_price_index != 0):
        candidate_price_index -= 1
        candidate_price = prices[candidate_price_index]
    return candidate_price, candidate_price_index


# Define Market Variables
solar_price = 0.3
off_peak_price = 10.1
mid_peak_price = 14.4
on_peak_price = 20.8

prices = [solar_price,off_peak_price,mid_peak_price,on_peak_price]


if __name__ == "__main__":
    # Read-in consumers' bids
    demand_matrix = aggDemandFiles()
    supply_matrix = -1*demand_matrix
    supply_matrix[supply_matrix < 0] = 0
    demand_matrix[demand_matrix < 0] = 0

    # Find clearing prices
    num_periods = demand_matrix.shape[2]
    clearing_prices = np.zeros(num_periods)

    for period in range(num_periods):
        demand_matrix_of_period = demand_matrix[:,:,period]
        supply_matrix_of_period = supply_matrix[:,:,period]
        clearing_prices[period],_ = findIntersect(np.sum(demand_matrix_of_period,axis=1),np.sum(supply_matrix_of_period,axis=1),prices)
    np.savetxt('clearing_prices.csv',clearing_prices)
