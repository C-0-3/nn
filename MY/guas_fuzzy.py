import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV dataset using pandas
csv_file = 'your_dataset.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Assuming the relevant column for validation is named 'value' (adjust accordingly)
data_values = data['value'].values  # Replace 'value' with the correct column name in your dataset

# Create a Gaussian function
def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# Calculate mean and standard deviation for Gaussian function
mean = np.mean(data_values)
sigma = np.std(data_values)

# Generate Gaussian model
gaussian_model = gaussian(data_values, mean, sigma)

# Check non-linearity: If the absolute difference between data and model is small, reject the dataset (linear).
if np.max(np.abs(data_values - gaussian_model)) < 0.5:
    print("Data is linear. Rejecting dataset for fuzzy training.")
else:
    print("Data is non-linear. Accepting dataset for fuzzy training.")
    accepted_data = data_values
    # Plot the original data vs Gaussian model
    plt.plot(data_values, label='Original Data')
    plt.plot(gaussian_model, label='Gaussian Model', linestyle='dashed')
    plt.legend()
    plt.show()



if 'accepted_data' in locals():
    # Define fuzzy variables
    data_range = np.arange(0, 11, 1)
    data_fuzzy = ctrl.Antecedent(data_range, 'data')

    # Membership functions: low, medium, high
    data_fuzzy['low'] = fuzz.trimf(data_range, [0, 0, 5])
    data_fuzzy['medium'] = fuzz.trimf(data_range, [0, 5, 10])
    data_fuzzy['high'] = fuzz.trimf(data_range, [5, 10, 10])

    # Define fuzzy output (for simplicity, let's assume a "decision" output)
    decision_range = np.arange(0, 11, 1)
    decision_fuzzy = ctrl.Consequent(decision_range, 'decision')

    # Membership functions for output
    decision_fuzzy['low'] = fuzz.trimf(decision_range, [0, 0, 5])
    decision_fuzzy['medium'] = fuzz.trimf(decision_range, [0, 5, 10])
    decision_fuzzy['high'] = fuzz.trimf(decision_range, [5, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(data_fuzzy['low'], decision_fuzzy['low'])
    rule2 = ctrl.Rule(data_fuzzy['medium'], decision_fuzzy['medium'])
    rule3 = ctrl.Rule(data_fuzzy['high'], decision_fuzzy['high'])

    # Create control system and simulation
    fuzzy_system = ctrl.ControlSystem([rule1, rule2, rule3])
    fuzzy_simulation = ctrl.ControlSystemSimulation(fuzzy_system)

    # Apply fuzzy logic to each accepted data point
    for point in accepted_data:
        fuzzy_simulation.input['data'] = point
        fuzzy_simulation.compute()
        print(f"Input: {point}, Fuzzy Decision: {fuzzy_simulation.output['decision']}")
