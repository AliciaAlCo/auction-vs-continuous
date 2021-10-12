
# -*- coding: utf-8 -*-
"""
Created in October 2021

@author: Eléa Prat

Finds the worst or best social welfare for a given set of bids in a continuous market with network constraints and block bids

From the paper "Auction-Based vs Continuous Clearing in Local Flexibility Markets with Block Bids" by A. A. Cobacho, E. Prat, D. V. Pombo and S. Chatzivasileiadis
"""

import pyomo.environ as pyo

import numpy as np
import pandas as pd


#%% Parameters and data files

setpoint_file = 'Setpoint5.csv'
network_file = 'network5bus.xlsx'
bids_file = 'Bids_energy_5.csv'

bigM = 1000

case = 'worst'# 'worst' to obtain the worst social welfare or 'best' to obtain the best social welfare
display_model = False
solver = 'gurobi'
display_solver = False
display_results = False

#%% Prepare data

Setpoint = pd.read_csv(open(setpoint_file, 'rb'),index_col=0) # Baseline injections at each nodes (negative for retrieval)

# Index for nodes
bus_df = pd.read_excel(open(network_file,'rb'),sheet_name='Bus',index_col=0)
nodes = list(bus_df.index)

# Index for branches
branch_df= pd.read_excel(open(network_file,'rb'),sheet_name='Branch',index_col=0)
lines = list(branch_df.index)

# Upload bids
bids_df = pd.read_csv(open(bids_file,'rb'),index_col=0)

# Create empty dataframes to contain the bids according to their type and direction
offers = bids_df.loc[(bids_df['Bid']=='Offer') & (bids_df['Block']=='No')]
offers_BB = bids_df.loc[(bids_df['Bid']=='Offer') & (bids_df['Block']!='No')]
requests = bids_df.loc[(bids_df['Bid']=='Request')]

bb_dict = {}

# Build dictionary to identify up and down offers of a block bid
for ID in bids_df.index:
    if bids_df.at[ID,'Bid'] == 'Offer' and bids_df.at[ID,'Block'] != 'No':
            if bids_df.at[ID,'Direction'] == 'Up':
                bb_dict[(bids_df.at[ID,'Block'],'Up')] = ID
                
            elif bids_df.at[ID,'Direction'] == 'Down':
                bb_dict[(bids_df.at[ID,'Block'],'Down')] = ID

n_ref = bus_df[bus_df['type'] == 3].index[0] # Return the id of the reference bus
bids_list = list(offers.index) + list(np.unique(offers_BB['Block']))

# Optimization problem
def sequence(binary_cstr):
    
    m = pyo.ConcreteModel()
    
    #%% Sets and Parameters
    
    # Sets (description in paper)
    m.R = pyo.Set(initialize = requests.index)
    m.B = pyo.Set(initialize = bids_list)
    m.O = pyo.Set(initialize = offers.index)
    m.K = pyo.Set(initialize = np.unique(offers_BB['Block']))
    m.I = pyo.Set(initialize = range(len(bids_list)))
    m.T = pyo.Set(initialize = Setpoint.index)
    m.N = pyo.Set(initialize = nodes)
    m.L = pyo.Set(initialize = lines)
    
    #%% Parameters
    
    # Prices (description in paper)
    m.lmbd_r = pyo.Param(m.R, initialize={r:requests.at[r,'Price'] for r in m.R})
    m.lmbd_o = pyo.Param(m.O, initialize={o:offers.at[o,'Price'] for o in m.O})
    m.lmbd_U_k = pyo.Param(m.K, initialize={k:offers_BB.at[bb_dict[(k,'Up')],'Price'] for k in m.K})
    m.lmbd_D_k = pyo.Param(m.K, initialize={k:offers_BB.at[bb_dict[(k,'Down')],'Price'] for k in m.K})
    
    # Quantity bid (description in paper)
    m.P_max_r = pyo.Param(m.R, initialize={r:requests.at[r,'Quantity'] for r in m.R})
    m.P_max_o = pyo.Param(m.O, initialize={o:offers.at[o,'Quantity'] for o in m.O})
    m.P_U_max_k = pyo.Param(m.K, initialize={k:offers_BB.at[bb_dict[(k,'Up')],'Quantity'] for k in m.K})
    m.P_D_max_k = pyo.Param(m.K, initialize={k:offers_BB.at[bb_dict[(k,'Down')],'Quantity'] for k in m.K})
    
    # Setpoint
    m.P_S = pyo.Param(m.N, m.T, initialize={(n,t):Setpoint.at[t,n] for n in m.N for t in m.T})
    
    # Lines parameters (description in paper)
    m.b = pyo.Param(m.L, initialize={l:1/branch_df.at[l,'X'] for l in m.L})
    m.Pl_max = pyo.Param(m.L, initialize={l:branch_df.loc[l,'Pmax'] for l in m.L})
    
    #%% Variables
    
    # Upper Level variables (description in paper)
    m.P_tot_r = pyo.Var(m.R, domain=pyo.NonNegativeReals)
    m.P_tot_o = pyo.Var(m.O, domain=pyo.NonNegativeReals)
    m.AR_tot = pyo.Var(m.K, domain=pyo.Binary)
    m.s = pyo.Var(m.I, m.B, domain=pyo.Binary)
    
    # Primal variables (description in paper)
    m.P_r = pyo.Var(m.I, m.R, domain=pyo.NonNegativeReals)
    m.P_o = pyo.Var(m.I, m.O, domain=pyo.NonNegativeReals)
    m.P_U_k = pyo.Var(m.I, m.K, domain=pyo.NonNegativeReals)
    m.P_D_k = pyo.Var(m.I, m.K, domain=pyo.NonNegativeReals)
    if binary_cstr == 'tight':
        m.AR = pyo.Var(m.I, m.K, domain=pyo.Binary)
    elif binary_cstr == 'relax':
        def b1(m,i,k):
            return (0,1)
        m.AR = pyo.Var(m.I, m.K, domain=pyo.NonNegativeReals, bounds=b1)
    m.delta_i = pyo.Var(m.I, m.N, m.T, domain=pyo.Reals)
    
    # Dual variables
    m.mu_up_i = pyo.Var(m.I, m.T, domain=pyo.Reals) # dual for constraint (4b)
    m.mu_down_i = pyo.Var(m.I, m.T, domain=pyo.Reals) # dual for constraint (4c)
    m.nu_i = pyo.Var(m.I, m.N, m.T, domain=pyo.Reals) # dual for constraint (4d)
    m.gamma_i = pyo.Var(m.I, m.T, domain=pyo.Reals) # dual for constraint (4e)
    m.pi_min_i = pyo.Var(m.I, m.L, m.T, domain=pyo.NonNegativeReals) # dual for constraint (4f) left
    m.pi_max_i = pyo.Var(m.I, m.L, m.T, domain=pyo.NonNegativeReals) # dual for constraint (4f) right
    m.rho_i = pyo.Var(m.I, m.R, domain=pyo.NonNegativeReals) # dual for constraint (4g)
    m.sigma_i = pyo.Var(m.I, m.O, domain=pyo.NonNegativeReals) # dual for constraint (4h)
    m.beta_up_i = pyo.Var(m.I, m.K, domain=pyo.Reals) # dual for constraint (4i)
    m.beta_down_i = pyo.Var(m.I, m.K, domain=pyo.Reals) # dual for constraint (4j)
    m.alpha_i = pyo.Var(m.I, m.K, domain=pyo.NonNegativeReals) # dual for constraint (4k) relaxed
    
    # Binary variables to linearize complementarity constraints of the lower-level
    m.bin_cc1_i = pyo.Var(m.I, m.R, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4g) left
    m.bin_cc2_i = pyo.Var(m.I, m.R, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4g) right
    m.bin_cc3_i = pyo.Var(m.I, m.K, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4k) relaxed left
    m.bin_cc4_i = pyo.Var(m.I, m.K, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4k) relaxed right
    m.bin_cc5_i = pyo.Var(m.I, m.O, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4h) left
    m.bin_cc6_i = pyo.Var(m.I, m.O, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4h) right
    m.bin_cc7_i = pyo.Var(m.I, m.L, m.T, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4f) left
    m.bin_cc8_i = pyo.Var(m.I, m.L, m.T, domain=pyo.Binary) # Associated with the complementarity constraint from constraint (4f) right
    
    #%%  Objective function
    
    obj = sum(m.P_tot_r[r] * m.lmbd_r[r] for r in m.R)
    obj -= sum(m.P_tot_o[o] * m.lmbd_o[o] for o in m.O)
    obj -= sum(m.AR_tot[k]*(m.P_U_max_k[k] * m.lmbd_U_k[k] + m.P_D_max_k[k] * m.lmbd_D_k[k]) for k in m.K)
    
    if case == 'worst': # Minimize social welfare
        m.obj_cost = pyo.Objective(expr= obj, sense = pyo.minimize)
    elif case == 'best': # Maximize social welfare
        m.obj_cost = pyo.Objective(expr= obj, sense = pyo.maximize)
        
    #%%  Constraints
    
    # Upper level
    
    # 3c
    def UL_P_r_def_rule(m,r):
        return m.P_tot_r[r] == sum(m.P_r[i,r] for i in m.I)
    # 3d
    def UL_P_o_def_rule(m,o):
        return m.P_tot_o[o] == sum(m.P_o[i,o] for i in m.I)
    # 3e
    def UL_AR_rule(m,k):
        return m.AR_tot[k] == sum(m.AR[i,k] for i in m.I)
    # 3g
    def UL_s_sum1_rule(m,i):
        return sum(m.s[i,b] for b in m.B) == 1
    # 3h
    def UL_s_sum2_rule(m,b):
        return sum(m.s[i,b] for i in m.I) == 1
    
    # Lower-level
    
    # Constraints primal
    
    # 4b
    def LL_bal_up_rule(m,i,t):
        rhs = sum(m.P_U_k[i,k] for k in m.K if offers_BB.at[bb_dict[(k,'Up')],'Time_target'] == t)
        rhs += sum(m.P_o[i,o] for o in m.O if (offers.at[o,'Direction'] == 'Up' and offers.at[o,'Time_target'] == t))
        lhs = sum(m.P_r[i,r] for r in m.R if (requests.at[r,'Direction'] == 'Up' and requests.at[r,'Time_target'] == t))
        return rhs == lhs
    # 4c
    def LL_bal_down_rule(m,i,t):
        rhs = sum(m.P_D_k[i,k] for k in m.K if offers_BB.at[bb_dict[(k,'Down')],'Time_target'] == t)
        rhs += sum(m.P_o[i,o] for o in m.O if (offers.at[o,'Direction'] == 'Down' and offers.at[o,'Time_target'] == t))
        lhs = sum(m.P_r[i,r] for r in m.R if (requests.at[r,'Direction'] == 'Down' and requests.at[r,'Time_target'] == t))
        return rhs == lhs
    # 4d
    def LL_nodal_bal_rule(m,i,n,t):
        rhs = Setpoint.at[t,n]
        # single offers
        rhs+= sum(m.P_o[j,o] for j in m.I for o in m.O if (j<=i and offers.at[o,'Direction'] == 'Up' and offers.at[o,'Time_target'] == t and offers.at[o,'Bus'] == n))
        rhs-= sum(m.P_o[j,o] for j in m.I for o in m.O if (j<=i and offers.at[o,'Direction'] == 'Down' and offers.at[o,'Time_target'] == t and offers.at[o,'Bus'] == n))
        # block bids
        rhs+= sum(m.P_U_k[j,k] for j in m.I for k in m.K if (j<=i and offers_BB.at[bb_dict[(k,'Up')],'Time_target'] == t and offers_BB.at[bb_dict[(k,'Up')],'Bus'] == n))
        rhs-= sum(m.P_D_k[j,k] for j in m.I for k in m.K if (j<=i and offers_BB.at[bb_dict[(k,'Down')],'Time_target'] == t and offers_BB.at[bb_dict[(k,'Down')],'Bus'] == n))
        # requests
        rhs-= sum(m.P_r[j,r] for j in m.I for r in m.R if (j<=i and requests.at[r,'Direction'] == 'Up' and requests.at[r,'Time_target'] == t and requests.at[r,'Bus'] == n))
        rhs+= sum(m.P_r[j,r] for j in m.I for r in m.R if (j<=i and requests.at[r,'Direction'] == 'Down' and requests.at[r,'Time_target'] == t and requests.at[r,'Bus'] == n))
        # line flow
        rhs-= sum(m.b[l]*(m.delta_i[i,n,t] - m.delta_i[i,branch_df.at[l,'To'],t]) for l in m.L if branch_df.at[l,'From'] == n)
        rhs-= sum(m.b[l]*(m.delta_i[i,n,t] - m.delta_i[i,branch_df.at[l,'From'],t]) for l in m.L if branch_df.at[l,'To'] == n)
        return rhs== 0
    # 4e
    def LL_ref_rule(m,i,t):
        return m.delta_i[i,n_ref,t] == 0
    # 4f left
    def LL_flow_min_rule(m,i,l,t):
        return m.b[l]*(m.delta_i[i,branch_df.loc[l,'From'],t] - m.delta_i[i,branch_df.loc[l,'To'],t]) >= - m.Pl_max[l]
    # 4f right
    def LL_flow_max_rule(m,i,l,t):
        return m.b[l]*(m.delta_i[i,branch_df.loc[l,'From'],t] - m.delta_i[i,branch_df.loc[l,'To'],t]) <= m.Pl_max[l]
    # 4g
    def LL_P_r_max_rule(m,i,r):
        return m.P_r[i,r] <= m.P_max_r[r] - sum(m.P_r[j,r] for j in m.I if j<i)
    # 4h
    def LL_P_o_max_rule(m,i,o):
        return m.P_o[i,o] <= sum(m.s[j,o] for j in m.I if j<=i) * (m.P_max_o[o] - sum(m.P_o[j,o] for j in m.I if j<i))
    # 4i
    def LL_P_U_k_rule(m,i,k):
        return m.P_U_k[i,k] == m.s[i,k] * m.AR[i,k] * m.P_U_max_k[k]
    # 4j
    def LL_P_D_k_rule(m,i,k):
        return m.P_D_k[i,k] == m.s[i,k] * m.AR[i,k] * m.P_D_max_k[k]
    
    # Constraints dual
    def LL_dual_P_r_rule(m,i,r):
        t = requests.at[r,'Time_target']
        n = requests.at[r,'Bus']
        lhs = - requests.at[r,'Price']
        if requests.at[r,'Direction'] == 'Up':
            lhs -= m.mu_up_i[i,t]
            lhs -= m.nu_i[i,n,t]
        elif requests.at[r,'Direction'] == 'Down':
            lhs -= m.mu_down_i[i,t]
            lhs += m.nu_i[i,n,t]
        lhs += m.rho_i[i,r]
        return lhs >= 0
    
    def LL_dual_P_o_rule(m,i,o):
        t = offers.at[o,'Time_target']
        n = offers.at[o,'Bus']
        lhs = offers.at[o,'Price']
        if offers.at[o,'Direction'] == 'Up':
            lhs += m.mu_up_i[i,t]
            lhs += m.nu_i[i,n,t]
        elif offers.at[o,'Direction'] == 'Down':
            lhs += m.mu_down_i[i,t]
            lhs -= m.nu_i[i,n,t]
        lhs += m.sigma_i[i,o]
        return lhs >= 0
    
    def LL_dual_P_U_k_rule(m,i,k):
        bb = bb_dict[(k,'Up')]
        t = offers_BB.at[bb,'Time_target']
        n = offers_BB.at[bb,'Bus']
        return m.mu_up_i[i,t] + m.nu_i[i,n,t] + m.beta_up_i[i,k] == 0
    
    def LL_dual_P_D_k_rule(m,i,k):
        bb = bb_dict[(k,'Down')]
        t = offers_BB.at[bb,'Time_target']
        n = offers_BB.at[bb,'Bus']
        return m.mu_down_i[i,t] - m.nu_i[i,n,t] + m.beta_down_i[i,k] == 0
    
    def LL_dual_AR_rule(m,i,k):
        lhs = m.lmbd_U_k[k] * m.P_U_max_k[k]
        lhs += m.lmbd_D_k[k] * m.P_D_max_k[k]
        lhs -= m.beta_up_i[i,k] * m.s[i,k] * m.P_U_max_k[k]
        lhs -= m.beta_down_i[i,k] * m.s[i,k] * m.P_D_max_k[k]
        lhs += m.alpha_i[i,k]
        return lhs >= 0
    
    def LL_dual_delta_rule(m,i,n,t):
        lhs = 0
        for l in m.L:
            if n == branch_df.loc[l,'From']:
                nl = branch_df.loc[l,'To']
                lhs += 1/branch_df.at[l,'X'] * (m.nu_i[i,nl,t] - m.nu_i[i,n,t] + m.pi_max_i[i,l,t] - m.pi_min_i[i,l,t])
            elif n == branch_df.loc[l,'To']:
                nl = branch_df.loc[l,'From']
                lhs += 1/branch_df.at[l,'X'] * (m.nu_i[i,nl,t] - m.nu_i[i,n,t] - m.pi_max_i[i,l,t] + m.pi_min_i[i,l,t])
        if n == n_ref:
            lhs += m.gamma_i[i,t]
        return lhs == 0
    
    # Complementarity constraints
    def LL_cc1_lhs_rule(m,i,r):
        return m.P_r[i,r] <= m.bin_cc1_i[i,r] * bigM
    
    def LL_cc1_rhs_rule(m,i,r):
        t = requests.at[r,'Time_target']
        n = requests.at[r,'Bus']
        lhs = - requests.at[r,'Price']
        if requests.at[r,'Direction'] == 'Up':
            lhs -= m.mu_up_i[i,t]
            lhs -= m.nu_i[i,n,t]
        elif requests.at[r,'Direction'] == 'Down':
            lhs -= m.mu_down_i[i,t]
            lhs += m.nu_i[i,n,t]
        lhs += m.rho_i[i,r]
        return lhs <= (1-m.bin_cc1_i[i,r]) * bigM
    
    def LL_cc2_lhs_rule(m,i,r):
        return (m.P_max_r[r] - sum(m.P_r[j,r] for j in m.I if j<i)) - m.P_r[i,r] <= m.bin_cc2_i[i,r] * bigM
    
    def LL_cc2_rhs_rule(m,i,r):
        return m.rho_i[i,r]  <= (1-m.bin_cc2_i[i,r]) * bigM
    
    def LL_cc3_lhs_rule(m,i,k):
        return m.AR[i,k] <= m.bin_cc3_i[i,k] * bigM
    
    def LL_cc3_rhs_rule(m,i,k):
        lhs = m.lmbd_U_k[k] * m.P_U_max_k[k]
        lhs += m.lmbd_D_k[k] * m.P_D_max_k[k]
        lhs -= m.beta_up_i[i,k] * m.s[i,k] * m.P_U_max_k[k]
        lhs -= m.beta_down_i[i,k] * m.s[i,k] * m.P_D_max_k[k]
        lhs += m.alpha_i[i,k]
        return lhs <= (1-m.bin_cc3_i[i,k]) * bigM
    
    def LL_cc4_lhs_rule(m,i,k):
        return 1 - m.AR[i,k] <= m.bin_cc4_i[i,k] * bigM
    
    def LL_cc4_rhs_rule(m,i,k):
        return m.alpha_i[i,k] <= (1-m.bin_cc4_i[i,k]) * bigM
    
    def LL_cc5_lhs_rule(m,i,o):
        return m.P_o[i,o] <= m.bin_cc5_i[i,o] * bigM
    
    def LL_cc5_rhs_rule(m,i,o):
        t = offers.at[o,'Time_target']
        n = offers.at[o,'Bus']
        lhs = offers.at[o,'Price']
        if offers.at[o,'Direction'] == 'Up':
            lhs += m.mu_up_i[i,t]
            lhs += m.nu_i[i,n,t]
        elif offers.at[o,'Direction'] == 'Down':
            lhs += m.mu_down_i[i,t]
            lhs -= m.nu_i[i,n,t]
        lhs += m.sigma_i[i,o]
        return lhs <= (1-m.bin_cc5_i[i,o]) * bigM
    
    def LL_cc6_lhs_rule(m,i,o):
        return sum(m.s[j,o] for j in m.I if j<=i) * (m.P_max_o[o] - sum(m.P_o[j,o] for j in m.I if j<i)) - m.P_o[i,o] <= m.bin_cc6_i[i,o] * bigM
    
    def LL_cc6_rhs_rule(m,i,o):
        return m.sigma_i[i,o] <= (1-m.bin_cc6_i[i,o]) * bigM
    
    def LL_cc7_lhs_rule(m,i,l,t):
        return m.b[l]*(m.delta_i[i,branch_df.loc[l,'From'],t] - m.delta_i[i,branch_df.loc[l,'To'],t]) + m.Pl_max[l] <= m.bin_cc7_i[i,l,t] * bigM
    
    def LL_cc7_rhs_rule(m,i,l,t):
        return m.pi_min_i[i,l,t] <= (1-m.bin_cc7_i[i,l,t]) * bigM
    
    def LL_cc8_lhs_rule(m,i,l,t):
        return m.Pl_max[l] - m.b[l]*(m.delta_i[i,branch_df.loc[l,'From'],t] - m.delta_i[i,branch_df.loc[l,'To'],t]) <= m.bin_cc8_i[i,l,t] * bigM
    
    def LL_cc8_rhs_rule(m,i,l,t):
        return m.pi_max_i[i,l,t] <= (1-m.bin_cc8_i[i,l,t]) * bigM
    
    m.UL_P_r_def = pyo.Constraint(m.R, rule=UL_P_r_def_rule)
    m.UL_P_o_def = pyo.Constraint(m.O, rule=UL_P_o_def_rule)
    m.UL_s_sum1 = pyo.Constraint(m.I, rule=UL_s_sum1_rule)
    m.UL_s_sum2 = pyo.Constraint(m.B, rule=UL_s_sum2_rule)
    m.UL_AR = pyo.Constraint(m.K, rule=UL_AR_rule)
    
    m.LL_bal_up = pyo.Constraint(m.I, m.T, rule=LL_bal_up_rule)
    m.LL_bal_down = pyo.Constraint(m.I, m.T, rule=LL_bal_down_rule)
    m.LL_nodal_bal = pyo.Constraint(m.I, m.N, m.T, rule=LL_nodal_bal_rule)
    m.LL_ref = pyo.Constraint(m.I, m.T, rule=LL_ref_rule)
    m.LL_flow_min = pyo.Constraint(m.I, m.L, m.T, rule=LL_flow_min_rule)
    m.LL_flow_max = pyo.Constraint(m.I, m.L, m.T, rule=LL_flow_max_rule)
    m.LL_P_r_max = pyo.Constraint(m.I, m.R, rule=LL_P_r_max_rule)
    m.LL_P_o_max = pyo.Constraint(m.I, m.O, rule=LL_P_o_max_rule)
    m.LL_P_U_k_max = pyo.Constraint(m.I, m.K, rule=LL_P_U_k_rule)
    m.LL_P_D_k_max = pyo.Constraint(m.I, m.K, rule=LL_P_D_k_rule)
    
    m.LL_dual_P_r = pyo.Constraint(m.I, m.R, rule=LL_dual_P_r_rule)
    m.LL_dual_P_o = pyo.Constraint(m.I, m.O, rule=LL_dual_P_o_rule)
    m.LL_dual_P_U_k = pyo.Constraint(m.I, m.K, rule=LL_dual_P_U_k_rule)
    m.LL_dual_P_D_k = pyo.Constraint(m.I, m.K, rule=LL_dual_P_D_k_rule)
    m.LL_dual_AR = pyo.Constraint(m.I, m.K, rule=LL_dual_AR_rule)
    m.LL_dual_delta = pyo.Constraint(m.I, m.N, m.T, rule=LL_dual_delta_rule)
    
    m.LL_cc1_lhs = pyo.Constraint(m.I, m.R, rule=LL_cc1_lhs_rule)
    m.LL_cc1_rhs = pyo.Constraint(m.I, m.R, rule=LL_cc1_rhs_rule)
    m.LL_cc2_lhs = pyo.Constraint(m.I, m.R, rule=LL_cc2_lhs_rule)
    m.LL_cc2_rhs = pyo.Constraint(m.I, m.R, rule=LL_cc2_rhs_rule)
    m.LL_cc3_lhs = pyo.Constraint(m.I, m.K, rule=LL_cc3_lhs_rule)
    m.LL_cc3_rhs = pyo.Constraint(m.I, m.K, rule=LL_cc3_rhs_rule)
    m.LL_cc4_lhs = pyo.Constraint(m.I, m.K, rule=LL_cc4_lhs_rule)
    m.LL_cc4_rhs = pyo.Constraint(m.I, m.K, rule=LL_cc4_rhs_rule)
    m.LL_cc5_lhs = pyo.Constraint(m.I, m.O, rule=LL_cc5_lhs_rule)
    m.LL_cc5_rhs = pyo.Constraint(m.I, m.O, rule=LL_cc5_rhs_rule)
    m.LL_cc6_lhs = pyo.Constraint(m.I, m.O, rule=LL_cc6_lhs_rule)
    m.LL_cc6_rhs = pyo.Constraint(m.I, m.O, rule=LL_cc6_rhs_rule)
    m.LL_cc7_lhs = pyo.Constraint(m.I, m.L, m.T, rule=LL_cc7_lhs_rule)
    m.LL_cc7_rhs = pyo.Constraint(m.I, m.L, m.T, rule=LL_cc7_rhs_rule)
    m.LL_cc8_lhs = pyo.Constraint(m.I, m.L, m.T, rule=LL_cc8_lhs_rule)
    m.LL_cc8_rhs = pyo.Constraint(m.I, m.L, m.T, rule=LL_cc8_rhs_rule)
    
    return m

#%% Solve

# First round: relax binary constraint
binary_cstr = 'relax'
m = sequence(binary_cstr)

if display_model == True:
    print('----- Model (relax binary constraint in lower-levels) -----')
    m.pprint()

opt = pyo.SolverFactory(solver)
opt.solve(m, tee=display_solver)

print('----- Results (relax binary constraint in lower-levels) -----')
if display_results == True:
    m.display()

obj_val = pyo.value(m.obj_cost)
print('Objective: {} Social Welfare: {}'.format(case,round(obj_val,3)))

print()

# Second round: reintroduce binary constraint
binary_cstr = 'tight'
m = sequence(binary_cstr)

if display_model == True:
    print('----- Model (with binary constraint in lower-levels) -----')
    m.pprint()

opt = pyo.SolverFactory(solver)
opt.solve(m, tee=display_solver)

print('----- Results (with binary constraint in lower-levels) -----')
if display_results == True:
    m.display()

obj_val = pyo.value(m.obj_cost)
print('Objective: {} Social Welfare: {}'.format(case,round(obj_val,3)))


