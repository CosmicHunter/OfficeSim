from torch.nn import Parameter
from torch.nn.functional import softmax, relu
import torch
import numpy as np
import torch.nn as nn

def mlp(input_dim, mlp_dims, last_relu=False):
    print("helllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllo")
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    print("model inside ",net)
    return net

class GraphConvNet(nn.Module):
    def __init__(self , self_state_dim , other_state_dim,wall_state_dim , obs_state_dim , X_dim , mlp_dims):
        super().__init__()
        print("Graph Conv NEt Initialized")
        self.self_state_dim = self_state_dim
        self.other_state_dim = other_state_dim
        self.wall_state_idx = self.self_state_dim + self.other_state_dim
        self.wall_state_dim = wall_state_dim
        self.obs_state_dim = obs_state_dim
        self.obs_state_idx = self.wall_state_idx + self.wall_state_dim
        
        print(f"wall_state_idx : {self.wall_state_idx}")
        print(f"obs_state_idx : {self.obs_state_idx}")

        self.num_layers = 2

        self.X_dim = X_dim
        self.final_state_dim = self.X_dim
        self.mlp1_dims = [64 , self.X_dim]
        self.mlp2_dims = [64 , self.X_dim]
        

        self.mlp1 = mlp(self.self_state_dim , self.mlp1_dims , last_relu = True)
        self.mlp2 = mlp(self.other_state_dim , self.mlp2_dims , last_relu = True)
        self.mlp3 = mlp(self.wall_state_dim , self.mlp2_dims , last_relu = True)
       
        self.W_A = Parameter(torch.randn(self.X_dim, self.X_dim))
        self.value_network = mlp(self.final_state_dim , mlp_dims)
        # 2 weight matrices for 2 layers of graphconvnet
        self.W_1 = Parameter(torch.randn(self.X_dim , self.X_dim))
        self.W_2 = Parameter(torch.randn(self.X_dim , self.X_dim))

    def get_similarity_matrix(self,  X):
        # we use embedded gaussian for similarity
        # matrix form is given by softmax (X , Wa , X_transpose)

        # X is of size (# num of batch , # no of agents , # X_dim)
        temp = torch.matmul(torch.matmul(X , self.W_A) , X.permute(0 , 2 , 1)) 
        normalized_A = softmax(temp , dim =2) # dim =2 is along the values (dim= 0 is batch , dim = 1 is no of nodes)
        return normalized_A    # size is (# batch , no of nodes , no of nodes)


    def forward(self ,state):
        self_state = state[: , 0 , : self.self_state_dim]
        self_state = nn.functional.normalize(self_state, p=2.0, dim=1, eps=1e-12, out=None)
        # print("self state : ",self_state)
        # print("self state size ",self_state.size())
        self_embedding = self.mlp1(self_state)      # size will be (batch size , size of embedding vector X_dim)
        other_agents_state = state[: , : , self.self_state_dim : self.wall_state_idx]
        other_agents_state = nn.functional.normalize(other_agents_state, p=2.0, dim=2, eps=1e-12, out=None)
        # print("other agents state : ",other_agents_state)
        # print("other agents state size ",other_agents_state.size())
        other_agents_embedding = self.mlp2(other_agents_state)   # size will be ( batch size , # num of other agents , size of embedding vector X_dim)
        # print("self embedding size : ", self_embedding.size())
        wall_state = state[: , 0 , self.wall_state_idx : self.wall_state_idx + self.wall_state_dim]
        wall_state = nn.functional.normalize(wall_state, p=2.0, dim=1, eps=1e-12, out=None)
        wall_state_embedding = self.mlp3(wall_state)
        obs_state = state[: , 0 , self.obs_state_idx : self.obs_state_idx + self.obs_state_dim]
        obs_state = nn.functional.normalize(obs_state, p=2.0, dim=1, eps=1e-12, out=None)
        obs_state_embedding = self.mlp3(obs_state)
        # print("wall embedding size ",wall_state_embedding.size())
        # print("obs embedding size " ,obs_state_embedding.size())
        # input matrix
        X = torch.cat([self_embedding.unsqueeze(dim = 1) , other_agents_embedding,wall_state_embedding.unsqueeze(dim=1),obs_state_embedding.unsqueeze(dim=1)]  , dim = 1)  # dimension of X ( num of batch , # no of agents , # X_dim)
        # print("size x" , X.size())
        normalized_A1 = self.get_similarity_matrix(X) # this is like a fancy adjacency matrix

        # layer propagation on 2 graph conv layers ( activation(AXW) )  W is the weight matrix
        hidden_state1 = relu(torch.matmul(torch.matmul(normalized_A1 , X) , self.W_1))  # shape [batch , num agent , X_dim]
        # print("h1 size" , hidden_state1.size())
        normalized_A2 = self.get_similarity_matrix(hidden_state1)   # shape is [batch , num agent , num agent]
        hidden_state2 = relu(torch.matmul(torch.matmul(normalized_A2 , hidden_state1) , self.W_2)) # shape is [batch , num agent , X_dim]
        # print("h2 size" , hidden_state2.size())
        # extract the agent feature from hidden state2
        # we are predicting for agent that has self state in the state passed , so we will pass this feature for one agent through mlp of [150, 100 , 100 , 1]
        state_feature = hidden_state2[: , 0 , :]  # shape of feature is [# batch , X_dim]
        # print("state_feature" , state_feature.size())
        value = self.value_network(state_feature)
        return value

