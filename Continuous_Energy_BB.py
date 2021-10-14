
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:31:09 2020
@author: emapr and aalarcon
"""

import pandas as pd
import numpy as np

from PTDF_check import PTDF_check
from itertools import combinations
from operator import add
import ast
import time

from BB_match import BB_match
from BB_match_WithoutNC import BB_match_WithoutNC

# PTDF check -> network_constraints = 1 (yes) / 0 (no)
def continuous_clearing_BB(all_bids,network_constraints):
    
    start_time = time.time()
    
    #%% Case data
    Setpoint_ini = pd.read_excel(open('Setpoint.xlsx', 'rb'),index_col=0)
    Setpoint = pd.DataFrame(columns = ['Time_target','Setpoint_P'])
    Setpoint.set_index('Time_target',inplace=True)
    setpoint = []
    for t in Setpoint_ini.index:
        setpoint = []
        for n in Setpoint_ini.columns:
            setpoint.append(Setpoint_ini.at[t,n])
        Setpoint.at[t,'Setpoint_P'] = setpoint
    
    # Initial Social Welfare and Flexibility Procurement Cost
    Social_Welfare = 0
    Flex_procurement = 0
    
    # Index for nodes
    bus = pd.read_excel(open('network33bus.xlsx', 'rb'),sheet_name='Bus',index_col=0)
    nodes = list(bus.index)
    
    lines_df = pd.read_excel(open('network33bus.xlsx', 'rb'),sheet_name='Branch',index_col=0)
    lines = list(lines_df.index)
       
    # Create empty dataframes to contain the bids that were not matched (order book)
    orderbook_offer = pd.DataFrame(columns = ['ID','Bus','Direction','Quantity','Price','Time_target','Time_stamp','Block'])
    orderbook_offer.set_index('ID',inplace=True)
    orderbook_request = pd.DataFrame(columns = ['ID','Bus','Direction','Quantity','Price','Time_target','Time_stamp','Temporary']) 
    orderbook_request.set_index('ID',inplace=True)
    # Create empty dataframes to contain the bids that were temporary matched
    orderbook_temporary = pd.DataFrame(columns = ['Offer','Offer Bus','Offer Block','Offer Price','Offer Time_stamp','Request','Request Bus','Request Price','Request Time_stamp','Direction','Quantity','Matching Price','Time_target'])
    
    #%% Function to match a new offer
    def continuous_clearing(new_bid, orderbook_request, orderbook_offer, orderbook_temporary, Setpoint, Social_Welfare, Flex_procurement):
   
        matches = pd.DataFrame(columns = ['Offer','Offer Bus','Offer Price','Offer Block','Request','Request Bus','Request Price','Direction','Quantity','Matching Price','Time_target','Social Welfare'])
    
    #%% Function to match an offer
        def matching(bid_type, Setpoint, bid, orderbook_request, orderbook_offer, orderbook_temporary,  matches, Social_Welfare, Flex_procurement):
                   
            #bid_type: new or old
            
            epsilon = 0.00001 # Tolerance
            status = 'no match' # Marker to identify if there was a match with unconditional requests or not (if so, the order book should be checked for new matches)
            flag = 'NaN' # initialize the output flag 
            
            time_target = bid.at['Time_target']
            Setpoint_t = Setpoint.at[time_target,'Setpoint_P']
            direction = bid.at['Direction']
            temporary = 'No'
            
            if bid_type == 'new':
                bid_nature = bid.at['Bid'] # Offer or request
            elif bid_type == 'old':
                bid_nature = 'Offer' 
                
            if bid_nature == 'Offer':
                offer_bus = nodes.index(bid.at['Bus'])
                offer_price = bid.at['Price']
                offer_quantity = bid.at['Quantity']
                offer_index = bid.name
                offer_time_stamp = bid.at['Time_stamp']
                offer_block = bid.at['Block']
        
                # Make sure that there are requests left to be matched
                if orderbook_request[(orderbook_request.Direction == direction) & (orderbook_request.Time_target == time_target)].empty:
                    flag = 'Empty orderbook'
                    orderbook_offer.loc[offer_index]=[nodes[offer_bus],direction,offer_quantity,offer_price,time_target,offer_time_stamp,offer_block]
                    orderbook_offer.sort_values(by=['Time_target','Price','Time_stamp'], ascending=[True,True,True], inplace=True) # Sort by price and by time submission and gather by time target
                    return Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary,  matches, flag, Social_Welfare, Flex_procurement
                
                # Else, list of requests to look into
                orderbook = orderbook_request
                
            elif bid_nature == 'Request':
                request_bus = nodes.index(bid.at['Bus'])
                request_price = bid.at['Price']
                request_quantity = bid.at['Quantity']
                request_index = bid.name
                request_time_stamp = bid.at['Time_stamp']
                
                # Make sure that there are offers left to be matched
                if orderbook_offer[(orderbook_offer.Direction == direction) & (orderbook_offer.Time_target == time_target)].empty:
                    flag = 'Empty orderbook'
                    orderbook_request.loc[request_index]=[nodes[request_bus],direction,request_quantity,request_price,time_target,request_time_stamp,temporary]
                    orderbook_request.sort_values(by=['Time_target','Price','Time_stamp'], ascending=[True,False,True], inplace=True) # Sort by price and by time submission and gather by time target
                    return Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary,  matches, flag, Social_Welfare, Flex_procurement
                
                # Else, list of requests to look into
                orderbook = orderbook_offer
            
            if bid_nature == 'Offer':
                if orderbook['Quantity'].sum() != orderbook_request['Quantity'].sum():
                    print ("Orderbook has not been updated properly")
     
            if bid_nature == 'Request':
                if orderbook['Quantity'].sum() != orderbook_offer['Quantity'].sum():
                    print ("Orderbook has not been updated properly")

            # Check matching with all the requests (in the same direction)
            for ID in orderbook.index:
                
                if ID not in orderbook.index:
                    continue
            
                if bid_nature == 'Offer':
                    if orderbook['Quantity'].sum() != orderbook_request['Quantity'].sum():
                        print ("Orderbook has not been updated properly")
     
                if bid_nature == 'Request':
                    if orderbook['Quantity'].sum() != orderbook_offer['Quantity'].sum():
                        print ("Orderbook has not been updated properly")
                
                # Check if there are bids to match with
                if orderbook.at[ID,'Direction'] == direction and orderbook.at[ID,'Time_target'] == time_target:
                #if orderbook.at[ID,'Direction'] == direction and orderbook.at[ID,'Time_target'] == time_target and offer_quantity != 0:
                    print ('---There is a possible match of {} with {}'.format(bid.name,ID))
                    
                    if bid_nature == 'Offer':
                        request_price = orderbook_request.at[ID,'Price']
                        request_index = ID
                    elif bid_nature == 'Request':
                        offer_price = orderbook_offer.at[ID,'Price']
                        offer_index = ID
                
                    # Make sure that the prices are matching
                    if offer_price <= request_price:
                        print ('Prices are matching')
                        if bid_nature == 'Offer':
                            request_bus = nodes.index(orderbook_request.at[ID,'Bus'])
                            request_index = ID
                            Offered = offer_quantity
                            Requested = orderbook_request.at[ID,'Quantity']
                            request_time_stamp = orderbook_request.at[ID,'Time_stamp']
                        elif bid_nature == 'Request':
                            offer_bus = nodes.index(orderbook_offer.at[ID,'Bus'])
                            offer_index = ID
                            Offered = orderbook_offer.at[ID,'Quantity']
                            Requested = request_quantity
                            offer_time_stamp = orderbook_offer.at[ID,'Time_stamp']
                            offer_block = orderbook_offer.at[ID,'Block']
                            
                        Quantity = min(Offered,Requested) # Initially, the maximum quantity that can be exchanged is the minimum of the quantities of the bids
                        
                        print ('Possible quantity exchanged: {}'.format(Quantity))
                        
    
                                                           
    #-----------------------#Check if the new bid is a block bid
                        if offer_block != 'No':
                            flag = 'Temporary match'
                            status = 'Temporary match'
                            temporary = 'Yes'
                            print ('The offer is a block bid')
                            
                            if bid_nature == 'Offer': 
                                orderbook_request.at[ID,'Temporary'] = 'Yes' # Mark the request as temporary matched
    
                            #Check if any part of the BB has been already matched with the same request  
                            a = 0
                            for Offer in orderbook_temporary.index:      
                                  if orderbook_temporary.at[Offer,'Offer'] == offer_index and orderbook_temporary.at[Offer,'Request'] == request_index:
                                      #print ('There is a match with this bid already')
                                      a = 1
                                      break                         
                            if a == 1: # Continue with the next bid
                                  continue
    
                            # 1. Add it to the orderbook (the offer quantity has to be always the initial)
                            
                            # 2. Create the temporary match and add it to the orderbook_temporary. The other orderbooks are not updated yet.
                            if request_time_stamp < offer_time_stamp:
                                matching_price = request_price
                            elif request_time_stamp > offer_time_stamp:
                                matching_price = offer_price
                            SW_single = (request_price-offer_price)*Quantity
                            orderbook_temporary = orderbook_temporary.append({'Offer':offer_index,'Offer Bus':nodes[offer_bus],'Offer Block':offer_block,'Offer Price':offer_price,'Offer Time_stamp':offer_time_stamp,'Request':request_index,'Request Bus':nodes[request_bus],'Request Price':request_price,'Request Time_stamp':request_time_stamp,'Direction':direction,'Quantity':Quantity,'Matching Price':matching_price, 'Time_target':time_target},ignore_index=True)
                            orderbook_temporary.sort_values(by=['Time_target','Direction','Matching Price'], ascending=[True,True,True], inplace=True) # Sort by price and by time submission and gather by time target
                            
                            # 3. Check if the whole BB has been matched (compare the quantity matched with the initial quantity)
                               #Check if any part of the BB has been already matched with the same request  
                            offer1_sum = 0
                            offer2_sum = 0
                            for i in orderbook_temporary.index:
                                if orderbook_temporary.at[i,'Offer'] == offer_index:
                                    offer1_sum += orderbook_temporary.at[i,'Quantity']
                                    print ('BB part {} is temporary matched with {}'.format(offer_index,orderbook_temporary.at[i,'Request']))
                                elif orderbook_temporary.at[i,'Offer Block'] == offer_block and orderbook_temporary.at[i,'Offer'] != offer_index:
                                    offer_index2 = orderbook_temporary.at[i,'Offer']
                                    offer2_sum += orderbook_temporary.at[i,'Quantity']
                                    print ('BB part {} is temporary matched with {}'.format(orderbook_temporary.at[i,'Offer'],orderbook_temporary.at[i,'Request']))
                            
                            if Offered <= offer1_sum: # If the quantity temporary matched is higher or equal to the offer_quantity this part of the BB could be matched
                                print ('BB part {} could be completely matched'.format(offer_index))
    
                                if offer_block not in orderbook_temporary.iloc[:,2].values: #If the other part has not been considered yet, the whole match can't happen
                                    print ('BB part {} is NOT temporary matched'.format(orderbook_temporary.at[i,'Offer']))
                                                   
                                else:
                                    i1 = 0
                                    for i in orderbook_offer[orderbook_offer['Block']==offer_block].index: # Look for the offers that are part of the BB
                                        i1 = 1
                                        if i in orderbook_temporary['Offer'].values:
                                            if i != offer_index: # Localize the other part of the BB in the orderbook                
                                                if orderbook_offer.at[i,'Quantity'] <= offer2_sum:
                                                    print ('BB part {} could also be completely matched'.format(i))
                                                    # 1. Match the ongoing part of the BB
                                                    # BB_match should return the matches that are considered for the BB
                                                    print ('Calling BB_match for the first time, with {}'.format(offer_index))
                                                    if network_constraints == 1:
                                                        BB_matches1 = BB_match(Setpoint_t, orderbook_temporary, offer_index, Offered, lines, nodes, direction, offer_bus, offer_price, offer_block)  
                                                    else:
                                                        BB_matches1 = BB_match_WithoutNC(Setpoint_t, orderbook_temporary, offer_index, Offered, lines, nodes, direction, offer_bus, offer_price, offer_block)  
    
                                                    if BB_matches1.empty:
                                                        print ('There is no feasible solution with the current temporary matches')
 
                                                    else:
                                                        # 2. Match the other part of the BB
                                                        print ('Calling BB_match for the second time, with {}'.format(offer_index2))
                                                        time_target2 = orderbook_offer.at[offer_index2,'Time_target']
                                                        Setpoint_t2 = Setpoint.at[time_target2,'Setpoint_P']
                                                        offer_quantity2 = orderbook_offer.at[offer_index2,'Quantity']
                                                        direction2 = orderbook_offer.at[offer_index2,'Direction']
                                                        offer_price2 = orderbook_offer.at[offer_index2,'Price']
                                                        if network_constraints == 1:
                                                            BB_matches2 = BB_match(Setpoint_t2, orderbook_temporary, offer_index2, offer_quantity2, lines, nodes, direction2, offer_bus, offer_price2, offer_block) 
                                                        else:
                                                            BB_matches2 = BB_match_WithoutNC(Setpoint_t2, orderbook_temporary, offer_index2, offer_quantity2, lines, nodes, direction2, offer_bus, offer_price2, offer_block) 
    
                                                        if BB_matches2.empty:
                                                            print ('There is no feasible solution with the current temporary matches')
                                                        
                                                        else:
                                                            print ('The whole block bid has been matched')
                                                            status = 'Match'
                                                            flag = 'Match'
                                                            # 3. Add the BB match to matches
                                                            BB_matches = pd.concat([BB_matches1,BB_matches2])
                                                            BB_matches = BB_matches.reset_index()
                                                            BB_matches = BB_matches.drop(['index'], axis=1)
                                                            matches = pd.concat([matches,BB_matches1,BB_matches2])
                                                            matches = matches.reset_index()
                                                            matches = matches.drop(['index'], axis=1)
                                                            #print (BB_matches)
    
                                                            # 4. Update the orderbooks
                                                                # 4.1 Remove the two parts of the offer from the orderbook_offer
                                                            if offer_index in orderbook_offer.index:
                                                                orderbook_offer = orderbook_offer.drop([offer_index], axis=0)
                                                            orderbook_offer = orderbook_offer.drop([offer_index2], axis=0)
                                                                # 4.2 Remove the temporary matches with this offer from the orderbook_temporary
                                                            for k in orderbook_temporary.index:
                                                                if orderbook_temporary.at[k,'Offer Block'] == offer_block:
                                                                    orderbook_temporary = orderbook_temporary.drop([k], axis=0)
                                                                # 4.3 Update the request in the orderbook_request
                                                            for m in BB_matches.index:
                                                                req = BB_matches.at[m,'Request']
                                                                req_quantity = BB_matches.at[m,'Quantity']
                                                                request_quantity = req_quantity
                                                                if req in orderbook_request.index.values: # If it is an old request, update/remove it
                                                                    orderbook_request.at[req,'Quantity'] -= req_quantity
                                                                    if orderbook_request.at[req,'Quantity'] < epsilon: # If the whole request has been matched
                                                                        orderbook_request = orderbook_request.drop([req], axis=0) # Remove it from the orderbook_request
                                                                        for j in orderbook_temporary[orderbook_temporary['Request']==req].index:
                                                                            orderbook_temporary = orderbook_temporary.drop([j], axis=0) # Remove its temporary matches
                                                                    else: # The whole request is not matched
                                                                        # Check if there is another temporary match with this request
                                                                        req_temporary = 0
                                                                        for j in orderbook_temporary.index:
                                                                            if req == orderbook_temporary.at[j,'Request']:
                                                                                if orderbook_temporary.at[j,'Quantity'] > orderbook_request.at[req,'Quantity']:
                                                                                    orderbook_temporary.at[j,'Quantity'] = orderbook_request.at[req,'Quantity']
                                                                                req_temporary += 1
                                                                        if req_temporary == 0: # There is no temporary matches with this request
                                                                            orderbook_request.at[req,'Temporary'] = 'No'
                                                              
                                                                    # Calculate the corresponding changes in the Setpoint
                                                            
                                                                Delta = [0] * len(Setpoint_t)
                                                                offer_bus_1 = nodes.index(BB_matches.at[m,'Offer Bus'])
                                                                request_bus_1 = nodes.index(BB_matches.at[m,'Request Bus'])
                                                                direction_1 = BB_matches.at[m,'Direction']
                                                                time_target_1 = BB_matches.at[m,'Time_target']
                                                                if direction_1 == 'Up':
                                                                    Delta[offer_bus_1]+=BB_matches.at[m,'Quantity']
                                                                    Delta[request_bus_1]-=BB_matches.at[m,'Quantity']
                                                                elif direction_1 == 'Down':
                                                                    Delta[offer_bus_1]-=BB_matches.at[m,'Quantity']
                                                                    Delta[request_bus_1]+=BB_matches.at[m,'Quantity']        
                                                                
                                                                # Modify the Setpoint and update the status marker
                                                                print ('Setpoint modification')
                                                                Setpoint_t = Setpoint.at[time_target_1,'Setpoint_P']
                                                                Setpoint.at[time_target_1,'Setpoint_P'] = list(map(add,Setpoint_t,Delta))
                                                                Setpoint_t = Setpoint.at[BB_matches.at[m,'Time_target'],'Setpoint_P']
                                                                status = 'match'

                                                                if bid_nature == 'Offer':
                                                                    orderbook = orderbook_request
                                                                    offer_quantity = 0

                                                                else:
                                                                    orderbook = orderbook_offer
                                                else:
                                                    print ('BUT BB part {} could NOT be completely matched'.format(i))
                                            
                                    if i1 == 0: # If the other part has not been considered yet
                                        print ('2.BUT the rest of the BB is not temporary matched')
                                        
                            else:
                                print ('BUT it could not be completely matched')                                          
                                
                        # If the bid it is not a BB
                        else:
                            print ('No BB involved')
                            
                            # 1. NETWORK CHECK
                            if network_constraints == 1:
                                if request_bus != offer_bus: # Network check, only if offer and request are not located at the same bus
                                    # Check for this match only
                                    Quantity = PTDF_check(Setpoint_t,Quantity,offer_bus,request_bus,direction) # Returns the maximum quantity that can be exchanged without leading to congestions
                                                   
                            if Quantity > epsilon: # Line constraints are respected
                                print ('THERE IS A MATCH')
                                print ('Quantity exchanged',Quantity)
                                flag = 'Match'
                                # The older bid sets the price
                                if request_time_stamp < offer_time_stamp:
                                    matching_price = request_price
                                elif request_time_stamp > offer_time_stamp:
                                    matching_price = offer_price
                                SW_single = (request_price-offer_price)*Quantity
                                # Flex_procurement += matching_price*Quantity
                                matches = matches.append({'Offer':offer_index,'Offer Bus':nodes[offer_bus], 'Offer Price':offer_price,'Offer Block':offer_block,'Request':request_index,'Request Bus':nodes[request_bus],'Request Price':request_price,'Matching Price':matching_price,'Direction':direction,'Quantity':Quantity, 'Time_target':time_target, 'Social Welfare':SW_single},ignore_index=True)               
        
                                # Calculate the corresponding changes in the Setpoint
                                Delta = [0] * len(Setpoint_t)
                                if direction == 'Up':
                                    Delta[offer_bus]+=Quantity
                                    Delta[request_bus]-=Quantity
                                elif direction == 'Down':
                                    Delta[offer_bus]-=Quantity
                                    Delta[request_bus]+=Quantity
                                    
                                # Modify the Setpoint and update the status marker
                                print ('Setpoint modification')
                                Setpoint.at[time_target,'Setpoint_P'] = list(map(add,Setpoint_t,Delta))
                                Setpoint_t = Setpoint.at[time_target,'Setpoint_P']
                                status = 'match'
                                
                                # Social welfare calculation 
                                Social_Welfare += (request_price-offer_price)*Quantity
                                
                                # Update the remaining quantities
                                if bid_nature == 'Offer':
                                    request_quantity = Requested - Quantity
                                    offer_quantity = Offered - Quantity
                                    orderbook_request.at[ID,'Quantity'] = request_quantity #Update orderbook_request
                                    
                                    # Check if the request matched has a previous temporary match with a BB
                                    if orderbook_request.at[ID,'Temporary'] == 'No':
                                        print ('Request {} was not temporary matched'.format(request_index))         
                                    else: 
                                        print ('Request {} was temporary matched'.format(request_index))  
                                        for i in orderbook_temporary[orderbook_temporary['Request']==request_index].index: # Update all the temporary matches that involve the request
                                            if orderbook_temporary.at[i,'Quantity'] > request_quantity:
                                                print ('The temporary match of {} and {} must be updated. Now the quantity matched is {}'.format(request_index,orderbook_temporary.at[i,'Offer'],request_quantity))
                                                orderbook_temporary.at[i,'Quantity'] = request_quantity
                                                if orderbook_temporary.at[i,'Quantity'] < epsilon:
                                                    print ('So it is completely cancelled')
                                                    orderbook_temporary = orderbook_temporary.drop([i], axis=0) # Remove the temporary match   
                                
                                    if ID in orderbook_request.index:
                                        if orderbook_request.at[ID,'Quantity'] < epsilon: # If the request was completely matched
                                            print ('Request completely matched')
                                            orderbook_request = orderbook_request.drop([ID], axis=0)
                                         
                                    orderbook = orderbook_request 
        
                                    if offer_quantity < epsilon: # If the offer was completely matched
                                        print ('Offer completely matched')
                                        flag = 'Match'
                                        status = 'Match'
                                        if bid_type == 'old': # In the case of checking the bids in the order book, the corresponding row must be dropped
                                            orderbook_offer = orderbook_offer.drop([offer_index], axis=0)
                                    
                                        return Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary,  matches, flag, Social_Welfare, Flex_procurement

                                elif bid_nature == 'Request':
                            
                                    request_quantity = Requested - Quantity
                                    orderbook_offer.at[ID,'Quantity'] = Offered - Quantity
                                    print ('Request quantity remaining',request_quantity)
                                    
                                    # Check if the request was temporary matched
                                    if request_index in orderbook_temporary.iloc[:,5].values:
                                        print ('Request {} was temporary matched'.format(request_index))
                                        for i in orderbook_temporary[orderbook_temporary['Request']==request_index].index: # Update all the temporary matches that involve the request
                                            if orderbook_temporary.at[i,'Quantity'] > request_quantity:
                                                print ('The temporary match of {} and {} must be updated. Now the quantity matched is {}'.format(request_index,orderbook_temporary.at[i,'Offer'],request_quantity))
                                                orderbook_temporary.at[i,'Quantity'] = request_quantity
                                                if orderbook_temporary.at[i,'Quantity'] < epsilon:
                                                    print ('So it is completely cancelled')
                                                    orderbook_temporary = orderbook_temporary.drop([i], axis=0) # Remove the temporary match   
                                   
                                    if orderbook_offer.at[ID,'Quantity'] < epsilon: # If the offer was completely matched
                                        print ('Offer completely matched')  
                                        orderbook_offer = orderbook_offer.drop([ID], axis=0)
                                      
                                    orderbook = orderbook_offer
                                      
                                    if request_quantity < epsilon: # If the request was completely matched
                                        print ('Request completely matched')     
                                        return Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary,  matches, flag, Social_Welfare, Flex_procurement
                            else:
                                flag = 'No match (congestions)'
                                print ('No match (congestions)')
                          
                    else:
                        flag = 'No match (price)'
                        break
                    
            if bid_nature == 'Offer':
                if offer_quantity > epsilon: # If the offer was not completely matched after trying all requests, update and order the book
                    orderbook_offer.loc[offer_index]=[nodes[offer_bus],direction,offer_quantity,offer_price,time_target,offer_time_stamp,offer_block]
                    orderbook_offer.sort_values(by=['Time_target','Price','Time_stamp'], ascending=[True,True,True], inplace=True) # Sort by price and by time submission and gather by time target
                orderbook = orderbook_request
                
            elif bid_nature == 'Request': 
                if request_quantity > epsilon: # If the request was not completely matched after trying all offers, update and order the book
                    orderbook_request.loc[request_index]=[nodes[request_bus],direction,request_quantity,request_price,time_target,request_time_stamp,temporary]
                    orderbook_request.sort_values(by=['Time_target','Price','Time_stamp'], ascending=[True,False,True], inplace=True) # Sort by price and by time submission and gather by time target
                orderbook = orderbook_offer
                
            return Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary, matches, flag, Social_Welfare, Flex_procurement
    
    #%% Check the power flows using PTDFs each time a bid is added
        
        Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary,  matches, flag, Social_Welfare, Flex_procurement = matching('new', Setpoint, new_bid, orderbook_request, orderbook_offer, orderbook_temporary,  matches, Social_Welfare, Flex_procurement)
        # If there was at least a match with an unconditional request, try again on older bids
        if status == 'match' and not orderbook_offer[(orderbook_offer.Direction == new_bid.at['Direction']) & (orderbook_offer.Time_target == new_bid.at['Time_target'])].empty:
            general_status = 'match'
            while general_status == 'match': # As long as previous offers are matching with unconditional requests, check for matches
                general_status = 'no match'
                for O in orderbook_offer.index:
                    old_offer = orderbook_offer.loc[O].copy()               
                    if old_offer['Time_target'] == new_bid.at['Time_target']:
                        if new_bid.name not in orderbook_offer.index:
                            break
                        else:
                            Setpoint, status, orderbook_request, orderbook_offer, orderbook_temporary, matches, flag_tp, Social_Welfare, Flex_procurement = matching('old',Setpoint, old_offer, orderbook_request, orderbook_offer, orderbook_temporary,  matches, Social_Welfare, Flex_procurement)
                        if status == 'match':
                            general_status = 'match'
        return matches, orderbook_request, orderbook_offer, orderbook_temporary,  Setpoint, flag, Social_Welfare, Flex_procurement
    
    All_Matches = []
    market_result = pd.DataFrame(columns = ['Offer','Offer Bus','Offer Block','Request','Request Bus','Direction','Quantity','Matching Price','Time_target','Social Welfare'])
    energy_volume = pd.DataFrame(columns = ['Time_target','Energy_volume', 'Social welfare'])
    energy_volume.set_index('Time_target',inplace=True)
    energy_volume_up = pd.DataFrame(columns = ['Time_target','Energy_volume'])
    energy_volume_up.set_index('Time_target',inplace=True)
    energy_volume_down = pd.DataFrame(columns = ['Time_target','Energy_volume'])
    energy_volume_down.set_index('Time_target',inplace=True)
    
    SocialW = pd.DataFrame(columns = ['Time_target','Social Welfare'])
    SocialW.set_index('Time_target',inplace=True)
    for t in Setpoint.index:
        SocialW.at[t,'Social Welfare'] = 0
    
    n=0
    print (all_bids)
    for b in all_bids.index:
        
        n+=1
        print ('')
        print('---------------- Betting round nb {} -----------------'.format(n))
        print ('')
        print('New bid: ({}, {}, {}, {}, {}, {}, {}, Block:{})'.format(b,all_bids.at[b,'Bid'],all_bids.at[b,'Bus'],all_bids.at[b,'Direction'],all_bids.at[b,'Quantity'],all_bids.at[b,'Price'],all_bids.at[b,'Time_target'],all_bids.at[b,'Block']))
        new_bid = all_bids.loc[b]
        matches, orderbook_request, orderbook_offer, orderbook_temporary,  Setpoint, flag, Social_Welfare, Flex_procurement = continuous_clearing(new_bid, orderbook_request, orderbook_offer, orderbook_temporary,  Setpoint, Social_Welfare, Flex_procurement)
        All_Matches.append([flag,matches])
        
        if not matches.empty:
            for i in matches.index:
                market_result = market_result.append({'Offer':matches.at[i,'Offer'],'Offer Bus':matches.at[i,'Offer Bus'],'Request Price':matches.at[i,'Request Price'],'Offer Block':matches.at[i,'Offer Block'],'Request':matches.at[i,'Request'],'Request Bus':matches.at[i,'Request Bus'],'Offer Price':matches.at[i,'Offer Price'],'Direction':matches.at[i,'Direction'],'Quantity':matches.at[i,'Quantity'],'Matching Price':matches.at[i,'Matching Price'],'Time_target':matches.at[i,'Time_target'],'Social Welfare':matches.at[i,'Social Welfare']},ignore_index=True)
                t = matches.at[i,'Time_target']
                SocialW.at[t,'Social Welfare'] += (matches.at[i,'Request Price'] - matches.at[i,'Offer Price'])*matches.at[i,'Quantity']
                
        print ('SOCIAL WELFARE', Social_Welfare)
        print ('')
        print ('State:',flag)
        #print ('REQUESTS')
        #print (orderbook_request.iloc[:,[1,2,3,4,6]])
        #print ('OFFERS')
        #print (orderbook_offer.iloc[:,[1,2,3,4,6]]) 
        #print ('TEMPORARY')
        #print (orderbook_temporary.iloc[:,[0,2,5,9]]) 
        #print ('MATCH')
        #print (matches.iloc[:,[0,3,4,8]])
           
    #market_result.sort_values(by=['Time_target','Direction'], ascending=[True,True], inplace=True)
    #print ('Market Result',market_result.iloc[:,[0,3,6,7,8]])
    #print ('Market Result',market_result)
    SocWel = 0
    
    total_volume = 0
    for t in Setpoint.index:
        vol = 0
        vol_up = 0
        vol_down = 0
        sol = 0
        for i in market_result.index:
            if market_result.at[i,'Time_target'] == t:
                vol += market_result.at[i,'Quantity']
                sol += (market_result.at[i,'Request Price'] - market_result.at[i,'Offer Price'])*market_result.at[i,'Quantity'] 
                if market_result.at[i,'Direction'] == 'Up':
                    vol_up += market_result.at[i,'Quantity']
                else:
                    vol_down += market_result.at[i,'Quantity']
        total_volume += vol
        energy_volume.at[t,'Energy_volume'] = vol
        energy_volume.at[t,'Social welfare'] = sol
        energy_volume_up.at[t,'Energy_volume'] = vol_up
        energy_volume_down.at[t,'Energy_volume'] = vol_down
        SocWel += SocialW.at[t,'Social Welfare']

    print ('Total Social Welfare', SocWel)
    print ('Total Volume',total_volume)
    
    if network_constraints == 1:
        print ('Network constraints considered')
    else:
        print ('Network constraints NOT considered')
    
    BB = 0
    for B in all_bids.index:
        if all_bids.at[B,'Block'] != 'No':
            BB = 1
    
    if BB == 1:
        print ('Block bids included')
    else:
        print ('Block bids NOT included')
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time: ", total_time)
    
    return (SocWel,total_volume,total_time)
    
    
    
    
    
    
    
    
    
    
    
