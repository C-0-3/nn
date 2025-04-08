import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create fuzzy variables and their respective membership functions using Triangle MF
def fuzzy_triangle_temp():
    temp = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')
    temp['Low'] = fuzz.trimf(temp.universe, [0, 0, 50])
    temp['Medium'] = fuzz.trimf(temp.universe, [20, 50, 80])
    temp['High'] = fuzz.trimf(temp.universe, [50, 100, 100])
    return temp

def fuzzy_triangle_pressure():
    pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'Pressure')
    pressure['Low'] = fuzz.trimf(pressure.universe, [0, 0, 50])
    pressure['High'] = fuzz.trimf(pressure.universe, [50, 100, 100])
    return pressure

def fuzzy_triangle_heating_power():
    heating_power = ctrl.Consequent(np.arange(0, 101, 1), 'Heating Power')
    heating_power['Low'] = fuzz.trimf(heating_power.universe, [0, 0, 50])
    heating_power['Medium'] = fuzz.trimf(heating_power.universe, [30, 50, 80])
    heating_power['High'] = fuzz.trimf(heating_power.universe, [50, 100, 100])
    return heating_power

def fuzzy_triangle_valve_opening():
    valve = ctrl.Consequent(np.arange(0, 101, 1), 'Valve Opening')
    valve['Low'] = fuzz.trimf(valve.universe, [0, 0, 50])
    valve['Medium'] = fuzz.trimf(valve.universe, [30, 50, 80])
    valve['High'] = fuzz.trimf(valve.universe, [50, 100, 100])
    return valve

# Create a function to process fuzzy rules based on user input
def create_fuzzy_rules():
    num_rules = int(input("Enter the number of rules: "))
    rules = []
    
    for i in range(num_rules):
        temp_val = input(f"Rule {i+1}: Temperature (Low/Medium/High) - ").capitalize()
        pressure_val = input(f"Rule {i+1}: Pressure (Low/High) - ").capitalize()
        heating_power_val = input(f"Rule {i+1}: Heating Power (Low/Medium/High) - ").capitalize()
        valve_val = input(f"Rule {i+1}: Valve Opening (Low/Medium/High) - ").capitalize()
        rules.append((temp_val, pressure_val, heating_power_val, valve_val))
    
    print("\nYour Fuzzy Rules:")
    for rule in rules:
        print(f"Rule: Temperature - {rule[0]}, Pressure - {rule[1]}, Heating Power - {rule[2]}, Valve Opening - {rule[3]}")

    return rules

# Function to map user input to fuzzy sets
def get_fuzzy_rule_conditions(rule, temp_trap, pressure_trap, heating_power_trap, valve_trap):
    temp_cond = temp_trap[rule[0]] if rule[0] in temp_trap.terms else None
    pressure_cond = pressure_trap[rule[1]] if rule[1] in pressure_trap.terms else None
    heating_power_cond = heating_power_trap[rule[2]] if rule[2] in heating_power_trap.terms else None
    valve_cond = valve_trap[rule[3]] if rule[3] in valve_trap.terms else None
    
    return temp_cond, pressure_cond, heating_power_cond, valve_cond

# Function to plot the fuzzy membership functions
def plot_membership_functions(temp_trap, pressure_trap, heating_power_trap, valve_trap):
    # Plot Temperature membership functions
    temp_trap.view()
    
    # Plot Pressure membership functions
    pressure_trap.view()
    
    # Plot Heating Power membership functions
    heating_power_trap.view()
    
    # Plot Valve Opening membership functions
    valve_trap.view()

# Main function to handle fuzzy inference system
def fuzzy_inference_system():
    # Initialize fuzzy variables with triangle membership functions
    temp_trap = fuzzy_triangle_temp()
    pressure_trap = fuzzy_triangle_pressure()
    heating_power_trap = fuzzy_triangle_heating_power()
    valve_trap = fuzzy_triangle_valve_opening()
    
    # Plot the membership functions
    plot_membership_functions(temp_trap, pressure_trap, heating_power_trap, valve_trap)
    
    # Get user-defined rules
    rules = create_fuzzy_rules()
    
    # Create list of fuzzy rules based on user input
    fuzzy_rules = []
    for rule in rules:
        temp_cond, pressure_cond, heating_power_cond, valve_cond = get_fuzzy_rule_conditions(rule, temp_trap, pressure_trap, heating_power_trap, valve_trap)
        
        # If all conditions are valid, create the rule
        if temp_cond and pressure_cond:
            if heating_power_cond:
                fuzzy_rules.append(ctrl.Rule(temp_cond & pressure_cond, heating_power_cond))
            if valve_cond:
                fuzzy_rules.append(ctrl.Rule(temp_cond & pressure_cond, valve_cond))
        else:
            print(f"Invalid rule: {rule}. Skipping.")
    
    # Create the fuzzy system
    system = ctrl.ControlSystem(fuzzy_rules)
    
    # Create a simulation object
    simulation = ctrl.ControlSystemSimulation(system)
    
    # Ask the user for input values
    temp_input = float(input("Enter Temperature (0-100): "))
    pressure_input = float(input("Enter Pressure (0-100): "))
    
    # Set the input values for fuzzy inference
    simulation.input[temp_trap.label] = temp_input
    simulation.input[pressure_trap.label] = pressure_input
    
    # Display fuzzified values
    temp_fuzzified = {term: fuzz.interp_membership(temp_trap.universe, temp_trap[term].mf, temp_input) 
                      for term in temp_trap.terms}
    pressure_fuzzified = {term: fuzz.interp_membership(pressure_trap.universe, pressure_trap[term].mf, pressure_input) 
                          for term in pressure_trap.terms}
    
    print("\nFuzzified Values:")
    print(f"Temperature: {temp_fuzzified}")
    print(f"Pressure: {pressure_fuzzified}")

    # Compute the fuzzy logic result
    simulation.compute()

    # Get the defuzzified output
    if heating_power_trap.label in simulation.output:
        heating_power_output_crisp = simulation.output[heating_power_trap.label]
    else:
        heating_power_output_crisp = "Not computed"
        
    if valve_trap.label in simulation.output:
        valve_output_crisp = simulation.output[valve_trap.label]
    else:
        valve_output_crisp = "Not computed"
    
    # Display the defuzzified outputs
    print(f"\nDefuzzified Heating Power: {heating_power_output_crisp}")
    print(f"Defuzzified Valve Output: {valve_output_crisp}")

    # Provide reasoning for the results
    print("\nReasoning for the results:")
    print(f"Based on the input Temperature: {temp_input} and Pressure: {pressure_input}, the system has computed the appropriate heating power and valve opening.")
    print(f"Heating power is determined to be {round(heating_power_output_crisp)} based on the current state.")

# Call the fuzzy inference system function
fuzzy_inference_system()
