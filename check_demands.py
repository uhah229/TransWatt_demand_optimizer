#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
path_to_file = os.path.join(cwd,'agent_demand_files')

files = os.listdir(path_to_file)
history_files = [file for file in files if 'demand_history' in file ]

plt.figure(figsize=(10,6))

for file in history_files:
    path_to_history = os.path.join(path_to_file,file)
    df = pd.read_csv(path_to_history)
    plt.plot(df)

plt.show()
    
