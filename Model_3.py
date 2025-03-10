# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:31:09 2024

@author: Diamantis
"""
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, maximize
from pyomo.opt.results import TerminationCondition
import logging
import time
import math

start_time = time.time()

#-------------Parameters-----------------------------------
    #Power production
Solar_capacity = 89e3 #kW
OffshoreWind_capacity = 2390e3 #kW
OnshoreWind_capacity = 119e3 #kW

#Prices
# methanol_price = 1.1 #EUR/kg
# H2_price = 5 #EUR/kg
eSAF_price = 1.6 #EUR/kg
naptha_price = 0.5 #EUR/kg
diesel_price = 0.6 #EUR/kg
eSAF_selling_interval = (24*364)/52 #t, (hour)
eSAF_selling_contract = 4800e3 #kg/per interval

#Conversion
MtHydrocarbon_ratio = 0.429 #methanol to hydrocarbon ratio
H2tHydrocarbon_ratio = 0.01 #hydrogen to hydrocarbon ratio
MtHydrocarbon_e_ratio = 0.3 #renewable energy ratio kWh/kg

eSAF_ratio = 0.90
naphta_ratio = 0.04
diesel_ratio = 0.06

#eSAF Operational parameters
eSAF_a = 250e6 #kg/year
MtK_capacity = 10.7e3 #kW
eSAFperHour_max = math.floor(eSAF_a / (24*365) / 0.8) #kg/h
methanol_import_max = 50e3 #kg
methanol_import_min = 30e3 #kg
print("Max eSAF production per hour: ", eSAFperHour_max)

#Methanol Operational parameters
methanol_e_ratio = (0.52 * 1.57 + 0.11) #renewable energy ratio kWh/kg
methanolSynthesis_capacity = 16e3 #kW
methanol_ramp_up_rate = 0.2 * methanolSynthesis_capacity  # kW/hour, ramp-up rate, TBD
methanol_ramp_down_rate = 0.2 * methanolSynthesis_capacity  # kW/hour, ramp-down rate, TBD
H2tM_ratio = 0.122 * 1.57 # kg/kg

#Storage
methanol_storage_capacity = 118100e3 #kg
H2_storage_capacity = 500e3 #kg
eSAF_storage_capacity = 50900e3 #kg

initial_methanol_storage = 0 #kg
initial_H2_storage = 0 #kg
initial_eSAF_storage = 0 #kg

#-------- Electrolysis----------------
etH2_ratio = 39.4 #kWh/kg
electrolyzer_capacity = 1.4e6 # kW, maximum capacity of the electrolyzer
electrolyzer_efficiency = 0.7  # Efficiency of the electrolyzer
ramp_up_rate = 0.5 * electrolyzer_capacity  # kW/hour, ramp-up rate of the electrolyzer
ramp_down_rate = 0.5 * electrolyzer_capacity  # kW/hour, ramp-down rate of the electrolyzer
min_operation_fraction = 0.05  # Minimum operation fraction for the electrolyzer
water_consumption_per_kg_hydrogen = 10  # Liters
water_price_per_liter = 0.05  # $/liter, example water price
initial_storage_hydrogen = 0
#-------
hydrogen_price = 5  # $/kg, price of hydrogen- fix price
oxygen_price = 1  # $/kg, price of oxygen- fix price
#-----
converter_eff = 0.98 #converter efficiency

total_capacity = electrolyzer_capacity



#---------------------------------------------- Define the path to your output file-----------------------------------
output_file = 'optimization_output_Model_3.txt'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Clear existing handlers to avoid duplication
if logger.hasHandlers():
    logger.handlers.clear()

# Create file handler
file_handler = logging.FileHandler(output_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

#---------------------------------------------------------------------Read data -----------------------------------------------
excel_file = 'input_data_2024_v1.xlsx'  # Replace with your actual file name
# Read capacity factor and electricity prices from Excel file
capacity_factor_df = pd.read_excel(excel_file, sheet_name='Capacity_Factor')
electricity_selling_prices_df = pd.read_excel(excel_file, sheet_name='Electricity_sell')
electricity_cost_df = pd.read_excel(excel_file, sheet_name='Electricity_cost')

# Convert to lists
hours = capacity_factor_df['Hour'].tolist()
technology_names = capacity_factor_df.columns[1:]  # Assumes the first column is 'Hour'
technology_capacity_factors = {tech: capacity_factor_df[tech].tolist() for tech in technology_names}
electricity_selling_price = electricity_selling_prices_df['Electricity_Prices_sell'].tolist()
electricity_cost_price = electricity_cost_df['Electricity_Prices_buy'].tolist()

# -----------------------------------Given data--------------------------------------- 
technology_capacities = {'Solar': Solar_capacity,'Offshore': OffshoreWind_capacity, 'Onshore': OnshoreWind_capacity}  #Capacities per technology
# Calculate potential power production based on capacity factors
power_production = {tech: [technology_capacities[tech] * cf for cf in technology_capacity_factors[tech]] for tech in technology_names}
power_available = {t: sum(power_production[tech][t] for tech in technology_names) for t in hours}


# ---------------------------------------------------------------------Optimization model--------------------------------------------------------------------------------------
model = ConcreteModel()

# Define decision variables dynamically
model.power_to_grid = Var(technology_names, hours, domain=NonNegativeReals)  # Power sold to the grid for each technology
model.power_to_electrolyzer = Var(technology_names, hours, domain=NonNegativeReals)  # Power used by the electrolyzer for each technology
model.stored_hydrogen = Var(hours, domain=NonNegativeReals)  # Hydrogen stored
model.hydrogen_sold = Var(hours, domain=NonNegativeReals)  # Hydrogen sold to off-taker
model.hydrogen_production = Var(hours, domain=NonNegativeReals)  # Hydrogen produced
model.oxygen_production = Var(hours, domain=NonNegativeReals)  # Oxygen produced
model.hydrogen_charge = Var(hours, domain=NonNegativeReals)  # Hydrogen charged to storage
model.hydrogen_discharge = Var(hours, domain=NonNegativeReals)  # Hydrogen discharged from storage
model.grid_electricity_purchase = Var(hours, domain=NonNegativeReals)  # Electricity purchased from the grid

# Step 5: Define the objective function
def objective_rule(model):

    return (
        sum(model.power_to_grid[tech, t] * electricity_selling_price[t] for tech in technology_names for t in hours)
        + sum(model.hydrogen_sold[t] * hydrogen_price for t in hours)
        + sum(model.oxygen_production[t] * oxygen_price for t in hours)
        + sum(model.grid_electricity_purchase[t] * electricity_cost_price[t] for t in hours)
        - sum(model.hydrogen_production[t] * water_consumption_per_kg_hydrogen * water_price_per_liter for t in hours)
    )

model.objective = Objective(rule=objective_rule, sense=maximize)

# ----------------------------------------- Define the constraints dynamically-------------------------------------------------

#_______________________________________Grid : ON/OFF_______________________________
# Constraint 17
def grid_electricity_purchase_rule(model, t):
    if(power_available[t] >= electrolyzer_capacity):
        return model.grid_electricity_purchase[t] == 0
    return model.grid_electricity_purchase[t] <= electrolyzer_capacity - power_available[t]

model.grid_electricity_purchase_constraint = Constraint(hours, rule=grid_electricity_purchase_rule)

def power_to_grid_rule(model, tech, t):
    if(power_available[t] <= electrolyzer_capacity):
        return model.power_to_grid[tech, t] == 0
    return model.power_to_grid[tech, t] <= power_production[tech][t] - model.power_to_electrolyzer[tech, t]

model.power_to_grid_constraint = Constraint(technology_names, hours, rule=power_to_grid_rule)


# Constraint 1: Power balance for each technology
def power_balance_rule(model, tech, t):
    return (model.power_to_grid[tech, t] + model.power_to_electrolyzer[tech, t]) <= power_production[tech][t]

model.power_balance = Constraint(technology_names, hours, rule=power_balance_rule)

# Constraint 2: Power used by the electrolyzer is bounded by its capacity
def electrolyzer_capacity_rule(model, t):
    return sum(model.power_to_electrolyzer[tech, t] for tech in technology_names) + model.grid_electricity_purchase[t] <= electrolyzer_capacity

model.electrolyzer_capacity = Constraint(hours, rule=electrolyzer_capacity_rule)


# Constraint 3: Minimum operation fraction for the electrolyzer
def electrolyzer_min_operation_rule(model, t):
    return (sum(model.power_to_electrolyzer[tech, t] for tech in technology_names) + model.grid_electricity_purchase[t]) >= min_operation_fraction * electrolyzer_capacity

model.electrolyzer_min_operation_constraint = Constraint(hours, rule=electrolyzer_min_operation_rule)


# Constraint 4: Ramp-up rates for the electrolyzer
def ramp_up_rule(model, t):
    if t == 0:
        return Constraint.Skip  # No ramp-up constraint for the first hour
    else:
        return (sum(model.power_to_electrolyzer[tech, t] for tech in technology_names) + model.grid_electricity_purchase[t]) - (sum(model.power_to_electrolyzer[tech, t-1] for tech in technology_names) + model.grid_electricity_purchase[t-1]) <= ramp_up_rate

model.ramp_up_constraint = Constraint(hours, rule=ramp_up_rule)

# Constraint 5: Ramp-down rates for the electrolyzer
def ramp_down_rule(model, t):
    if t == 0:
        return Constraint.Skip  # No ramp-down constraint for the first hour
    else:
        return (sum(model.power_to_electrolyzer[tech, t-1] for tech in technology_names) + model.grid_electricity_purchase[t-1]) - (sum(model.power_to_electrolyzer[tech, t] for tech in technology_names) + model.grid_electricity_purchase[t]) <= ramp_down_rate

model.ramp_down_constraint = Constraint(hours, rule=ramp_down_rule)


# Constraint 11: Hydrogen storage balance
def hydrogen_storage_rule(model, t):
    if t == 0:
        return model.stored_hydrogen[t] == initial_storage_hydrogen + model.hydrogen_charge[t] - model.hydrogen_discharge[t]
    else:
        return model.stored_hydrogen[t] == model.stored_hydrogen[t-1] + model.hydrogen_charge[t] - model.hydrogen_discharge[t]

model.hydrogen_storage = Constraint(hours, rule=hydrogen_storage_rule)

#Constraint 12 : Hydrogen production
def hydrogen_production_rule(model, t):
    return model.hydrogen_production[t] == (
        (sum(model.power_to_electrolyzer[tech, t] for tech in technology_names) +
        model.grid_electricity_purchase[t]) * electrolyzer_efficiency) / etH2_ratio

model.hydrogen_production_constraint = Constraint(hours, rule=hydrogen_production_rule)


# Constraint 13: Oxygen production 
def oxygen_production_rule(model, t):
    return model.oxygen_production[t] == model.hydrogen_production[t] * 8

model.oxygen_production_constraint = Constraint(hours, rule=oxygen_production_rule)

# Constraint 14: Hydrogen charge is equal to hydrogen production
def hydrogen_charge_rule(model, t):
    return model.hydrogen_charge[t] == model.hydrogen_production[t]

model.hydrogen_charge_constraint = Constraint(hours, rule=hydrogen_charge_rule)

# Constraint 15: Hydrogen discharge is equal to sold and this used from saf
def hydrogen_discharge_rule(model, t):
    return model.hydrogen_discharge[t] == model.hydrogen_sold[t]

model.hydrogen_discharge_constraint = Constraint(hours, rule=hydrogen_discharge_rule)

# Constraint 16: Hydrogen storage capacity
def hydrogen_storage_capacity_rule(model, t):
    return model.stored_hydrogen[t] <= H2_storage_capacity

model.hydrogen_storage_capacity = Constraint(hours, rule=hydrogen_storage_capacity_rule)

# Constraint 17: Hydrogen sold limit
def hydrogen_sold_rule(model, t):
    return model.hydrogen_sold[t] >= 0

model.hydrogen_sold_constraint = Constraint(hours, rule=hydrogen_sold_rule)




# ---------------------------------------------- Solve the model
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# Step 13: Extract and display results
if results.solver.termination_condition == TerminationCondition.optimal:
    # Extract values from the optimized model
    solar_power = [technology_capacities['Solar'] * 1e-3 * cf for cf in technology_capacity_factors['Solar']]
    offshore_power = [technology_capacities['Offshore'] * 1e-3 * cf for cf in technology_capacity_factors['Offshore']]
    onshore_power = [technology_capacities['Onshore'] * 1e-3 * cf for cf in technology_capacity_factors['Onshore']]
    total_power_available = [power_available[t] * 1e-3 for t in hours]
    power_to_grid = {(tech, t): model.power_to_grid[tech, t].value * 1e-3 for tech in technology_names for t in hours}  # I sell to the grid only the surplus of RES
    power_to_electrolyzer = {(tech, t): model.power_to_electrolyzer[tech, t].value * 1e-3 for tech in technology_names for t in hours} 
    stored_hydrogen = [model.stored_hydrogen[t].value for t in hours]
    hydrogen_sold = [model.hydrogen_sold[t].value for t in hours]
    hydrogen_production = [model.hydrogen_production[t].value for t in hours]
    oxygen_production = [model.oxygen_production[t].value for t in hours]
    water_consumption = [model.hydrogen_production[t].value * water_consumption_per_kg_hydrogen for t in hours]
    hydrogen_storage_charge = [model.hydrogen_charge[t].value for t in hours]
    grid_electricity_purchase = [model.grid_electricity_purchase[t].value * 1e-3 for t in hours]


        # Print sum of all variables over all hours

    total_power_to_grid = sum(model.power_to_grid[tech, t].value for tech in technology_names for t in hours)
    total_power_to_electrolyzer = sum(model.power_to_electrolyzer[tech, t].value for tech in technology_names for t in hours)
    final_stored_hydrogen = model.stored_hydrogen[hours[-1]].value
    total_hydrogen_sold = sum(model.hydrogen_sold[t].value for t in hours)
    total_hydrogen_production = sum(model.hydrogen_production[t].value for t in hours)
    total_oxygen_production = sum(model.oxygen_production[t].value for t in hours)
    total_water_consumption = sum(model.hydrogen_production[t].value * water_consumption_per_kg_hydrogen for t in hours)
    total_hydrogen_storage_charge = sum(model.hydrogen_charge[t].value for t in hours)
    total_hydrogen_storage_discharge = sum(model.hydrogen_discharge[t].value for t in hours)
    total_grid_electricity_purchase = sum(model.grid_electricity_purchase[t].value for t in hours)
    
    logger.info('__________________________________________________________________________________________')
    logger.info("Objective Function (Profit): %.2f Euros", model.objective())
    logger.info('Total Power Sold to Grid: %.2f KW', total_power_to_grid)
    logger.info('Total Power Used by Electrolyzer: %.2f KW', total_power_to_electrolyzer)
    logger.info('Stored Hydrogen at the end of the horizon: %.2f Kg', final_stored_hydrogen)
    logger.info('Total Hydrogen Sold: %.2f Kg', total_hydrogen_sold)
    logger.info('Total Hydrogen Production: %.2f Kg', total_hydrogen_production)
    logger.info('Total Oxygen Production: %.2f Kg', total_oxygen_production)
    logger.info('Total Water Consumption: %.2f Liters', total_water_consumption)
    logger.info('Total Hydrogen Storage Charge: %.2f Kg', total_hydrogen_storage_charge)
    logger.info('Total Grid Electricity Purchase: %.2f KW', total_grid_electricity_purchase)


    # Define the Excel file path
    results_file = 'optimization_results_Model_3.xlsx'

    # Ensure xlsxwriter is installed: pip install xlsxwriter

    # Initialize a writer to save multiple sheets
    with pd.ExcelWriter(results_file, engine='xlsxwriter') as writer:

        # First sheet: Technology power and electricity data
        technology_power_df = pd.DataFrame({'Hour': hours})
        technology_power_df['Solar_Power'] = solar_power
        technology_power_df['Offshore_Power'] = offshore_power
        technology_power_df['Onshore_Power'] = onshore_power
        technology_power_df['Total_Power_Available'] = total_power_available

        # Add power data for each technology dynamically
        for tech in technology_names:
            technology_power_df[f'Power_to_grid_{tech}'] = [power_to_grid[(tech, t)] for t in hours]
            technology_power_df[f'Power_to_electrolyzer_{tech}'] = [power_to_electrolyzer[(tech, t)] for t in hours]

        technology_power_df['Grid_to_electrolyzer'] = grid_electricity_purchase

        # Save technology power data to first sheet
        technology_power_df.to_excel(writer, sheet_name='Technology_Power', index=False)
    
        # Third sheet: Hydrogen production, oxygen production, and water consumption
        hydrogen_data_df = pd.DataFrame({'Hour': hours})
        hydrogen_data_df['Hydrogen_production'] = hydrogen_production
        hydrogen_data_df['Oxygen_production'] = oxygen_production
        hydrogen_data_df['Water_consumption'] = water_consumption
    
        # Save hydrogen data to third sheet
        hydrogen_data_df.to_excel(writer, sheet_name='Electrolysis_Data', index=False)
    
        # Fourth sheet: Hydrogen storage, charge, discharge, and sold data
        hydrogen_storage_df = pd.DataFrame({'Hour': hours})
        hydrogen_storage_df['Stored_hydrogen'] = stored_hydrogen
        hydrogen_storage_df['Hydrogen production'] = hydrogen_storage_charge
        hydrogen_storage_df['Hydrogen_sold'] = hydrogen_sold
    
        # Save hydrogen storage data to fourth sheet
        hydrogen_storage_df.to_excel(writer, sheet_name='Hydrogen_Data', index=False)
    
    # Log the successful save
    logger.info("Results saved to: %s", results_file)

else:
    logger.info("The optimization problem failed to converge. Check your model and constraints.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time/60} mins")

