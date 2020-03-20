#!/usr/bin/env python3

import os

def clean_up(cwd):
	path_to_agent_demand = os.path.join(cwd,'agent_demand_files')
	f_list = os.listdir(path_to_agent_demand)
	[os.remove(f) for f in os.path.join(path_to_agent_demand,f_list)]
