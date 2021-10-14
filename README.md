# auction-vs-continuous
Code for operating auction-based and continuous local flexibility markets considering block bids and network constraints.

## Case Study files
 * **network33bus.xlsx**: Network data.
 * **Bids_energy_33.xlsx**: Flexibility bids submitted (offers and requests).
 * **Setpoint.xlsx**: Initial setpoint of the grid.

## Market clearing 
The script **Scenarios_generation.py** calls the rest of the algorithms. It operates the auction-based and continuous clearing models for several scenarios with and without considering block bids and network constraints. Its ouputs (total social welfare, energy volume traded and computational time) are stored in the data frame **results**.
  * **arrival_creation.py** creates random arrival sequences for the bids in         **Bids_energy_33.xlsx**.
  * **Continuous_Energy_BB.py** is the continuous clearing model, that uses **PTDF_check.py** to check network constraints and **BB_match.py** and **BB_match_WithoutNC.py** to match block bids. 
  *  **Auction_Energy_BB.py** and **Auction_Energy_WithoutNC.py** are the auction-based clearing models that trade block bids with and without network constraints.



  
