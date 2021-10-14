# auction-vs-continuous
Code for operating auction-based and continuous local flexibility markets considering block bids and network constraints.

## Case Study files
 * **network33bus.xlsx**: Network data.
 * **Bids_energy_33.xlsx**: Flexibility bids submitted (offers and requests).
 * **Setpoint.xlsx**: Initial setpoint of the grid.

## Market clearing 
The script **Scenarios_generation.py** calls the rest of the algorithms.
  1* **arrival_creation.py** creates random arrival sequences for the bids in         **Bids_energy_33.xlsx**.
  2* **


# "Bids_energy_33.xlsx" containts all the bids (offers and requests) submitted for our study case.

# Running file "Scenarios_generation" implies generating a random arrival sequence of the bids and operating the continuous and auction-based LFM with and without considering block bids and network constraints. As output we obtain the total social welfare, energy volume traded and computational time.

  
