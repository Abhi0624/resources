import numpy as np
import random
from math import log, exp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        probabilities = torch.sigmoid(x)  # Treating as probabilities for MDL
        x = self.fc3(probabilities)
        return x, probabilities

def build_neural_network(input_dim):
    return NeuralNetwork(input_dim)

def train_neural_network(model, X_train, y_train, epochs=50, batch_size=16):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

def neural_network_mdl_score(network, data):
    num_nodes = len(network)
    node_to_index = {node: idx for idx, node in enumerate(network)}
    num_states = {node: np.unique(data[:, node_to_index[node]]).size for node in network}
    score = 0.0
    
    for node in network:
        parents = network[node]
        parent_indices = [node_to_index[parent] for parent in parents]
        node_index = node_to_index[node]
        
        if not parents:
            continue
        
        X = data[:, parent_indices]
        y = data[:, node_index]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = build_neural_network(len(parent_indices))
        train_neural_network(model, X_train, y_train)
        
        model.eval()
        with torch.no_grad():
            _, probabilities = model(torch.tensor(X_val, dtype=torch.float32))
        
        probabilities = probabilities.numpy().flatten()
        log_likelihood = np.sum(np.log(probabilities))
        
        num_parent_combinations = np.prod([num_states[parent] for parent in parents])
        model_length = (num_states[node] - 1) * num_parent_combinations
        score += log_likelihood + 0.5 * model_length * log(data.shape[0])
    
    return score

def k2_algorithm(node_order, data, max_parents=5):
    network = {node: [] for node in node_order}
    
    for node in node_order:
        best_score = neural_network_mdl_score(network, data)
        while len(network[node]) < max_parents:
            # Step 1: Select nodes before Xi and not in Pa(Xi)
            candidates = list(set(node_order[:node_order.index(node)]) - set(network[node]))
            if not candidates:
                break
            
            best_candidate = None
            for candidate in candidates:
                network[node].append(candidate)
                score = neural_network_mdl_score(network, data)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                network[node].remove(candidate)
                
            if best_candidate:
                network[node].append(best_candidate)
            else:
                break
    
    return network

def simulated_annealing(initial_order, data, initial_temp, cooling_rate, min_temp, max_iterations):
    current_order = initial_order[:]
    current_network = k2_algorithm(current_order, data)
    current_score = neural_network_mdl_score(current_network, data)
    current_energy = -current_score  # Step 2: Convert score to energy
    best_order = current_order[:]
    best_energy = current_energy
    
    temperature = initial_temp
    iteration_counter = 0
    
    while temperature > min_temp and iteration_counter < max_iterations:
        new_order = current_order[:]
        i, j = random.sample(range(len(new_order)), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]  # Swap two nodes
        
        new_network = k2_algorithm(new_order, data)
        new_score = neural_network_mdl_score(new_network, data)
        new_energy = -new_score  # Step 2: Convert score to energy
        
        # Step 3: Compare energies and decide acceptance
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = exp(-delta_energy / temperature)
        
        if acceptance_prob > random.random():
            current_order = new_order[:]
            current_energy = new_energy
            iteration_counter = 0  # Reset iteration counter when accepting new parent set
            if new_energy < best_energy:
                best_order = new_order[:]
                best_energy = new_energy
        else:
            iteration_counter += 1  # Increment iteration counter when not accepting new parent set
        
        # Step 4: Cooling schedule
        temperature *= cooling_rate
    
    return best_order, -best_energy

# Example usage with stock price time series data
# Assuming stock_prices is a 2D numpy array where each column represents a stock and each row represents a time step
stock_prices = np.random.rand(100, 5)  # Dummy stock price dataset with 100 time steps and 5 stocks
time_steps = 3  # Example number of time steps

# Create a list of nodes representing stocks at different time steps
initial_order = [(stock, time) for time in range(time_steps) for stock in range(stock_prices.shape[1])]
initial_temp = 1.0
cooling_rate = 0.95
min_temp = 0.001
max_iterations = 1000

# Flatten the data into a format suitable for the algorithm
flattened_data = np.hstack([stock_prices[:, np.newaxis] for _ in range(time_steps)]).reshape(-1, stock_prices.shape[1] * time_steps)

best_order, best_score = simulated_annealing(initial_order, flattened_data, initial_temp, cooling_rate, min_temp, max_iterations)
print("Best Node Order:", best_order)
print("Best Score:", best_score)