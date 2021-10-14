# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 09:39:41 2021

@author: aalarcon
"""

import pandas as pd

from arrival_creation import arrival_creation
from Continuous_Energy_BB import continuous_clearing_BB
from Auction_Energy_BB import auction_BB_NC
from Auction_Energy_WithoutNC import auction_BB_WNC

import time

start_time = time.time()

results = pd.DataFrame(columns = ['Social_Welfare_BB_NC','Volume_BB_NC','Time_BB_NC','Social_Welfare_SB_NC','Volume_SB_NC','Time_SB_NC','Social_Welfare_BB_WNC','Volume_BB_WNC','Time_BB_WNC','Social_Welfare_SB_WNC','Volume_SB_WNC','Time_SB_WNC','Social_Welfare_BB_NC_A','Volume_BB_NC_A','Time_BB_NC_A','Social_Welfare_SB_NC_A','Volume_SB_NC_A','Time_SB_NC_A','Social_Welfare_BB_WNC_A','Volume_BB_WNC_A','Time_BB_WNC_A','Social_Welfare_SB_WNC_A','Volume_SB_WNC_A','Time_SB_WNC_A'])

for i in range(0,1):
    # Creation of random arrival order
    all_bids_BB,all_bids_SB = arrival_creation()
     
    # network_constraints = 1 (yes) / 0 (no)
    # CONTINUOUS
    # With BB and NC
    social_welfare_BB_NC,energy_volume_BB_NC,time_total_BB_NC = continuous_clearing_BB(all_bids_BB,1)
    # Continuous with SB and NC
    social_welfare_SB_NC,energy_volume_SB_NC,time_total_SB_NC = continuous_clearing_BB(all_bids_SB,1)
    # Continuous with BB
    social_welfare_BB,energy_volume_BB,time_total_BB = continuous_clearing_BB(all_bids_BB,0)
    # Continuous with SB
    social_welfare_SB,energy_volume_SB,time_total_SB = continuous_clearing_BB(all_bids_SB,0)
    
    # AUCTION-BASED
    # With BB and NC
    social_welfare_BB_NC_A,energy_volume_BB_NC_A,time_total_BB_NC_A = auction_BB_NC(all_bids_BB)
    # With SB and NC
    social_welfare_SB_NC_A,energy_volume_SB_NC_A,time_total_SB_NC_A = auction_BB_NC(all_bids_SB)
     # With BB
    social_welfare_BB_A,energy_volume_BB_A,time_total_BB_A = auction_BB_WNC(all_bids_BB)
    # With SB
    social_welfare_SB_A,energy_volume_SB_A,time_total_SB_A = auction_BB_WNC(all_bids_SB)

    results.loc[i]=[social_welfare_BB_NC,energy_volume_BB_NC,time_total_BB_NC,social_welfare_SB_NC,energy_volume_SB_NC,time_total_SB_NC,social_welfare_BB,energy_volume_BB,time_total_BB,social_welfare_SB,energy_volume_SB,time_total_SB,social_welfare_BB_NC_A,energy_volume_BB_NC_A,time_total_BB_NC_A,social_welfare_SB_NC_A,energy_volume_SB_NC_A,time_total_SB_NC_A,social_welfare_BB_A,energy_volume_BB_A,time_total_BB_A,social_welfare_SB_A,energy_volume_SB_A,time_total_SB_A]

end_time = time.time()
simulation_time = end_time - start_time
print("Time: ", simulation_time)
