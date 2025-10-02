import torch
import geospaNN
import numpy as np
import ipdb

# 1. Define the Friedman's function, and specify the dimension of input covariates.
def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

p = 5; funXY = f5

# 2. Set the parameters for the spatial process.
sigma = 1
phi = 3/np.sqrt(2)
tau = 0.01
rho = 0.8
theta = torch.tensor([sigma, phi, tau, rho])
# theta2 

# 3. Set the hyperparameters of the data.
#n = 1000            # Size of the simulated sample.
n = 100            # Size of the simulated sample.
nn = 10             # Neighbor size used for NNGP.
batch_size = 50     # Batch size for training the neural networks.

# Simulate and split the data
# 1. Simulate the spatially correlated data with spatial coordinates randomly sampled on a [0, 10]^2 squared domain.
torch.manual_seed(2024)

#X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10], time_range = 10, spatio_temporal = True)

# Full GP (without nugget, for now)
X, Y, coord, cov, corerr = geospaNN.Simulation(
    n=n, 
    p=p, 
    neighbor_size=nn, 
    fx= funXY, 
    theta = theta[:3], 
    range=[0, 10], 
    time_range = 10, 
    spatio_temporal = False,
    isNNGP = True
)

# 2. Order the spatial locations by max-min ordering.
X, Y, coord, _ = geospaNN.spatial_order(X, Y, coord, method = 'max-min')

# 3. Build the nearest neighbor graph, as a torch_geometric.data.Data object.
data = geospaNN.make_graph(X, Y, coord, nn)

# 4. Split data into training, validation, testing sets.
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=nn, test_proportion=0.2)

# Compose the mlp structure and train
# 1. Define the mlp structure (torch.nn) to use.            
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

# 2.Define the NN-GLS corresponding model.
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))

# model = geospaNN.nnglst(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))

# 3.Define the NN-GLS training class with learning rate and tolerance.
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)

# 4.Train the model.
training_log = nngls_model.train(data_train, data_val, data_test, Update_init = 10, Update_step = 10)

train_estimate = model.estimate(data_train.x)
test_predict = model.predict(data_train, data_test)