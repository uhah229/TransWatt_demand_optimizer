#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import os


class DemandOptimizer:
    """
    Agent demand curve optimizer class
    """
    def __init__(self, agent_dict,agent_id):
        def W_arr_converter(self):
            if (self.W_arr <= self.curr_t + 1.5) and (self.W_arr > self.curr_t):
                W_arr_converted = np.zeros(6)
                W_arr_converted[int(4 * (self.W_arr - self.curr_t) - 1)] = 1
                self.W_arr = W_arr_converted
            else:
                self.W_arr = np.zeros(6)

        self.price_list = agent_dict["price_list"]
        self.curr_t = agent_dict["curr_t"]
        self.all_opts = agent_dict["all_opts"]
        self.P_water = agent_dict["P_water"]
        self.vol_water = agent_dict["vol_water"]
        self.water_usage = agent_dict["water_usage"]
        self.tank_area = agent_dict["tank_area"]
        self.curr_temp_water = agent_dict["curr_temp_water"]
        self.W_arr = agent_dict["W_arr"]
        W_arr_converter(self)
        self.P_air = agent_dict["P_air"]
        self.ground_area = agent_dict["ground_area"]
        self.height = agent_dict["height"]
        self.curr_temp_air = agent_dict["curr_temp_air"]
        self.slope_air = agent_dict["slope_air"]
        self.slope_water = agent_dict["slope_water"]
        self.desire_temp_air = agent_dict["desire_temp_air"]
        self.desire_temp_water = agent_dict["desire_temp_water"]
        self.alpha_water = agent_dict["alpha_water"]
        self.alpha_air = agent_dict["alpha_air"]
        self.alpha_cost = agent_dict["alpha_cost"]
        self.baseload = agent_dict["baseload"]
        self.solar = agent_dict["solar"]
        # self.agent_id = agent_dict["agent_id"]
        self.agent_id = agent_id


    def get_cost(self,opt, P, mod_price):
        """
        Function to get cost of electricity for an appliance:

            Returns: Total cost of enery from this appliance
        """
        # 0.25 hours per time step
        return sum(np.array(opt)*np.array(mod_price)*P*0.25)

    def get_tempout(self):
        """Temperature outside function using a sinusoidal curve with noise:

            Returns:
                A list of outside temperture.
        """
        time = np.arange(0,24*4,1)
        temp_out_daily = 5+ 5*np.sin(2*np.pi/96*time)#+np.random.normal(scale=0.5, size = len(time))
        if self.curr_t < 6:
            index = int((self.curr_t+18)*4) # + 24 hours - 6 hours
        else:
            index = int((self.curr_t-6)*4)
        indices = np.arange(index,index+6) % 96
        return temp_out_daily[indices]

    def get_water_temp(self, opt, temp_out):
        """
        Function to retrieve water temperature at the tank:

            Args:
                C: Heat capacity of water [kj/kg]
                U: Coefficient of heat transfer (energy/(time*area*temp)) [W/m2K]
                temp_inflow: Inflow temperature of water into the tank [deg C]

            Returns:
                Temperature of the water tank
        """
        C=4.186
        U=7.9
        temp_inflow = 13
        delta_t = 3600/4 # -- 1 hour (3600s)
        m = self.vol_water*1000 #volume of water * density [kg]
        scen_temp = np.zeros(6)
        temp_t = self.curr_temp_water
        for i in range(0,len(opt)):
            I_t = opt[i]
            w_t = self.W_arr[i]
            temp_out_t = temp_out[i] + 10 # added arbritrary temperature from outside
            temp_t = ((self.vol_water-self.water_usage*w_t)*(temp_t+273.15) + self.water_usage*w_t*(temp_inflow+273.15))/self.vol_water \
                    +(self.P_water*I_t-U/1000*self.tank_area*(temp_t-temp_out_t))*delta_t/(m*C) - 273.15

            scen_temp[i]= temp_t

            # changed /
            if scen_temp[i] > self.desire_temp_water:
                scen_temp[i] = self.desire_temp_water

        return scen_temp

    def get_house_temp(self, opt, temp_out):
        """
        Function to retrieve temperture of house:

            Args:
                C: Heat capacity of air [kj/kg]
                U: Coefficient of heat transfer (energy/(time*area*temp)) [W/m2K]

            Returns:
                Temperture of house
        """
        U=0.04
        C=1
        m = 1.225*self.ground_area*self.height
        A = self.ground_area + np.sqrt(self.ground_area) * self.height * 4
        delta_t = 3600/4 # -- 1 hour (3600s)
        scen_temp = np.zeros(len(opt))
        temp_t = self.curr_temp_air
        for i in range(0,len(opt)):
            I_t = opt[i]
            temp_out_t = temp_out[i]

            temp_t = temp_t + (self.P_air*I_t*0.6-U/1000*A*(temp_t-temp_out_t))*delta_t/(m*C)

            scen_temp[i]= temp_t
        return scen_temp

    def get_discomfort(self, scen_temperature, desire_temp, slope):
        """
        Function for discomfort based on temperature:

            Returns: The sum of thermal discomforts cost by the indoor temp
        """
        return sum(slope*abs(desire_temp-np.array(scen_temperature)))

    def get_total_objective(self, total_discomfort, total_cost, alpha_discomfort, alpha_cost):
        """
        Function for calculating total objective function:
        """
        return alpha_discomfort*total_discomfort + alpha_cost*total_cost

    def get_opt_sche(self, app, mod_price):

        """
        Function to retrieve optimal schedule:

            Returns list of optimal schedule
        """

        J_vector = np.zeros(len(self.all_opts))
        discomfort_vector = np.zeros(len(self.all_opts))
        cost_vector = np.zeros(len(self.all_opts))

        for i,opt in self.all_opts.iterrows():
            if app == 1:
                total_cost  = self.get_cost(opt, self.P_air, mod_price)
                temp_out = self.get_tempout()
                temp_indoor = self.get_house_temp(opt, temp_out)
                discomfort = self.get_discomfort(temp_indoor, self.desire_temp_air, self.slope_air)
                total_J = self.get_total_objective(discomfort, total_cost, self.alpha_air, self.alpha_cost)
                discomfort_vector[i] = discomfort
                cost_vector[i] = total_cost
                J_vector[i]=total_J

            if app == 2:
                total_cost  = self.get_cost(opt, self.P_water, mod_price)
                temp_out = self.get_tempout()
                temp_intank = self.get_water_temp(opt,temp_out)
                discomfort = self.get_discomfort(temp_intank, self.desire_temp_water, self.slope_water)
                total_J = self.get_total_objective(discomfort, total_cost, self.alpha_water, self.alpha_cost)
                discomfort_vector[i] = discomfort
                cost_vector[i] = total_cost
                J_vector[i]=total_J

        df = self.all_opts.copy()
        df['discomfort'] = discomfort_vector
        df['cost'] = cost_vector
        df['totalJ'] = J_vector
        best_idx = df.totalJ.idxmin()
        return df.iloc[best_idx, 0:6]

    def get_solar_power(self):
        time = np.arange(0,24*4,1)
        capacity = 11.34 * 0.2 # [kW] Tesla Solar Panel Rating x capacity factor
        sinusoid = capacity*np.sin(2*np.pi/96*time)+np.random.normal(scale=0.05, size = len(time))
        sinusoid[sinusoid<0] = 0

        # plt.plot(time,sinusoid)
        # plt.show()

        if self.curr_t < 6:
            index = int((self.curr_t+18)*4) # + 24 hours - 6 hours
        else:
            index = int((self.curr_t-6)*4)
        indices = np.arange(index,index+6) % 96
        return sinusoid[indices]

    def update_curr_temps(self,best_schedule_air,best_schedule_water,temp_out):
        temp_out_t = temp_out[0]
        temp_t = self.curr_temp_water
        w_t = self.W_arr[0]
        C=4.186
        U=7.9
        temp_inflow = 13
        delta_t = 3600/4 # -- 1 hour (3600s)
        m = self.vol_water*1000 #volume of water * density [kg]
        self.curr_temp_water = ((self.vol_water-self.water_usage*w_t)*(temp_t+273.15) + self.water_usage*w_t*(temp_inflow+273.15))/self.vol_water \
                +(self.P_water*best_schedule_water[0]-U/1000*self.tank_area*(temp_t-temp_out_t+10))*delta_t/(m*C) - 273.15

        U=0.04
        C=1
        m = 1.225*self.ground_area*self.height
        A = self.ground_area + np.sqrt(self.ground_area) * self.height * 4
        temp_t = self.curr_temp_air

        self.curr_temp_air += (self.P_air*best_schedule_air[0]*0.6-U/1000*A*(temp_t-temp_out_t))*delta_t/(m*C)


    def get_response(self):
        """
        Generates demand curve response based on price list input:

            Returns:
                The demand curve response

        """
        response = np.zeros((4,6))
        possible_prices = np.array([0.3,10.1,14.4,20.8])
        for t in range(6):
            for j,p in enumerate(possible_prices):
                mod_price = self.price_list.copy()
                mod_price[t] = p
                best_schedule_air = self.get_opt_sche(1,mod_price)
                best_schedule_water = self.get_opt_sche(2,mod_price)
                total_needed = best_schedule_air[t] * self.P_air + best_schedule_water[t] * self.P_water + self.baseload

                response[j,t] = total_needed

        if self.solar:
            response -= self.get_solar_power()
        # print("Current water temp:",self.curr_temp_water)
        # print("Current air temp:",self.curr_temp_air)
        best_schedule_air = self.get_opt_sche(1,self.price_list)
        best_schedule_water = self.get_opt_sche(2,self.price_list)

        air_next_opt = best_schedule_air[0]
        water_next_opt = best_schedule_water[0]

        # update temperatures
        self.update_curr_temps(best_schedule_air,best_schedule_water,self.get_tempout())
        next_timestep_demand = np.array([(self.baseload +
            self.P_air*air_next_opt + self.P_water*water_next_opt -
            self.get_solar_power()[0] * self.solar)]) # kWh

        
        # save agent demand + history files
        cwd = os.getcwd()
        output_dir = os.path.join(cwd,'agent_demand_files')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savetxt('agent_demand_files/agent{0:03d}_demand.csv'.format(self.agent_id),response,delimiter=',')
        with open('agent_demand_files/agent{0:03d}_demand_history.csv'.format(self.agent_id),'a') as f:
            np.savetxt(f,next_timestep_demand,delimiter=',')
        with open('agent_demand_files/agent{0:03d}_watertemp_history.csv'.format(self.agent_id),'a') as f:
            np.savetxt(f,np.array([self.curr_temp_water]))
        with open('agent_demand_files/agent{0:03d}_airtemp_history.csv'.format(self.agent_id),'a') as f:
            np.savetxt(f,np.array([self.curr_temp_air]))


        return self.curr_temp_air, self.curr_temp_water, next_timestep_demand

if __name__ == "__main__":
    agent_dict = {
        'price_list':[13.4,13.4,13.4,9.4,9.4,9.4],
        'curr_t' : 6,
        'all_opts' : pd.read_csv('binaries.txt',index_col = False, header = None, delimiter = ' '),
        'P_water' : 4,
        'vol_water' : 0.1514164,
        'water_usage' : 65.11/5/1000,
        'tank_area' : 3.9381,
        'curr_temp_water' : 60,
        'W_arr' : 7.25, #24 hour clock
        'P_air' : 1,
        'ground_area' : 72,
        'height' : 5,
        'curr_temp_air' : 22,
        'slope_air' : 0.4,
        'slope_water' : 1,
        'desire_temp_air' : 25,
        'desire_temp_water' : 60,
        'alpha_water' : 0.5,
        'alpha_air' : 1.5,
        'alpha_cost' : 1,
        'baseload' : 0.3,
        'solar': False,
        'agent_id': 1
    }
    dem_opt = DemandOptimizer(agent_dict)
    res = dem_opt.get_response() # demand matrix
