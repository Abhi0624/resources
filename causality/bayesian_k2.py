import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork

# Generate synthetic time series data
np.random.seed(42)
data = np.random.randn(100, 3)  # 100 time points, 3 variables

# Helper function to generate a neighboring solution by swapping two nodes in the order
def generate_neighbor(order):
    new_order = order[:]
    i, j = random.sample(range(len(order)), 2)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order

# Neural network model for predicting a node's values given its parents
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Helper function to prepare time-lagged data for a given node and its parents
def prepare_lagged_data(data, node, parents, lag):
    X, y = [], []
    for t in range(lag, len(data)):
        features = []
        for parent in parents:
            features.append(data[t-lag, parent])
        X.append(features)
        y.append(data[t, node])
    return np.array(X), np.array(y)

# Helper function to calculate the log-likelihood using neural networks for time-lagged data
def log_likelihood_nn_ts(data, node, parents, lag):
    if not parents:
        variance = np.var(data[lag:, node])
    else:
        X, y = prepare_lagged_data(data, node, parents, lag)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        model = SimpleNN(len(parents))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).numpy().flatten()

        residuals = y - predictions
        variance = np.var(residuals)

    return -0.5 * len(data) * np.log(2 * np.pi * variance)

# Helper function to calculate the MDL score using neural networks for time-lagged data
def mdl_score_nn_ts(data, node, parents, lag):
    ll = log_likelihood_nn_ts(data, node, parents, lag)
    num_params = sum(p.numel() for p in SimpleNN(len(parents)).parameters())
    mdl = ll - 0.5 * num_params * np.log(len(data))
    return mdl

# K2 algorithm with given node order using MDL score with neural networks for time-lagged data
def k2_algorithm_nn_ts(data, node_order, max_parents, lag):
    num_nodes = data.shape[1]
    best_parents = {i: [] for i in range(num_nodes)}
    best_scores = {i: mdl_score_nn_ts(data, i, best_parents[i], lag) for i in range(num_nodes)}

    for node in node_order:
        current_parents = best_parents[node]
        current_score = best_scores[node]
        improvement = True

        while improvement:
            improvement = False
            best_new_score = current_score
            best_new_parent = None

            for potential_parent in range(num_nodes):
                if potential_parent not in current_parents:
                    new_parents = current_parents + [potential_parent]
                    if len(new_parents) <= max_parents:
                        new_score = mdl_score_nn_ts(data, node, new_parents, lag)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_parent = potential_parent

            if best_new_parent is not None:
                current_parents.append(best_new_parent)
                current_score = best_new_score
                improvement = True

        best_parents[node] = current_parents
        best_scores[node] = current_score

    return best_parents

# Simulated Annealing for optimizing the node order for time-lagged data
def simulated_annealing_nn_ts(data, initial_order, max_parents, initial_temp, cooling_rate, max_iter, min_temp, lag):
    current_order = initial_order[:]
    current_score = sum(mdl_score_nn_ts(data, node, k2_algorithm_nn_ts(data, current_order, max_parents, lag)[node], lag) for node in current_order)
    best_order = current_order[:]
    best_score = current_score
    temp = initial_temp

    while temp > min_temp:
        for _ in range(max_iter):
            new_order = generate_neighbor(current_order)
            new_score = sum(mdl_score_nn_ts(data, node, k2_algorithm_nn_ts(data, new_order, max_parents, lag)[node], lag) for node in new_order)

            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < math.exp(-delta_score / temp):
                current_order = new_order
                current_score = new_score

                if current_score < best_score:
                    best_order = current_order
                    best_score = current_score

        temp *= cooling_rate

    return best_order

# Define the parameters
initial_order = list(range(data.shape[1]))
max_parents = 2
initial_temp = 100.0
cooling_rate = 0.95
max_iter = 100
min_temp = 1e-3
lag = 1  # Time lag to consider

# Run Simulated Annealing to find the best node order for time-lagged data
best_order = simulated_annealing_nn_ts(data, initial_order, max_parents, initial_temp, cooling_rate, max_iter, min_temp, lag)

# Run K2 algorithm with the best node order using MDL score with neural networks for time-lagged data
best_parents = k2_algorithm_nn_ts(data, best_order, max_parents, lag)

# Construct the Bayesian Network from the best parent sets
model = BayesianNetwork()
for node in range(data.shape[1]):
    for parent in best_parents[node]:
        model.add_edge(parent, node)

print("Best node order:", best_order)
print("Best parent sets:", best_parents)
print("Learned Bayesian Network structure:", model.edges())
