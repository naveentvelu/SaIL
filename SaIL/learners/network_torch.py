import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import DEVICE

class _SupervisedRegressionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class SupervisedRegressionNetwork():
    def __init__(self, params):    
        self.initialized=False
        self.output_size = params['output_size']
        self.input_size = params['input_size']
        self.learning_rate = params['learning_rate']
        self.batch_size =  params['batch_size']
        self.training_epochs = params['training_epochs']
        self.display_step = params['display_step']
        self.seed_val = params['seed_val']

    def initialize(self):
        if not self.initialized:
            torch.random.manual_seed(self.seed_val)
            self.net = _SupervisedRegressionNetwork(self.input_size, self.output_size).to(DEVICE)
            self.initialized = True
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.learning_rate)
            self.loss_function = nn.MSELoss()
    def create_network(self):
        pass

    def train(self, database):
        random.seed(0)
        avg_loss = 0
        for epoch_idx in range(self.training_epochs):
            random.shuffle(database)
            total_batch = int(len(database)/self.batch_size)
            avg_loss = 0
            for idx in range(total_batch):
                batch_x, batch_y = self._get_next_batch(database, idx)
                loss = self._single_pass(batch_x, batch_y)
                # TODO: Fix this weird loss statistics
                avg_loss += loss / total_batch
            if epoch_idx % self.display_step == 0:
                print ("epoch:", '%04d' % (epoch_idx + 1), "cost=", \
              "{:.9f}".format(np.sqrt(avg_loss)))
        print("Optimization Finished")
        return np.sqrt(avg_loss)

    def get_loss(self, features, label):
        pass

    def get_heuristic(self, features):
        x = torch.unsqueeze(self._to_tensor(features), 0)
        with torch.no_grad():
            out = self.net(x)
        return self._to_numpy(out).item()

    def save_params(self, file_name):
        torch.save(self.net.state_dict(), file_name)
        print("Model saved in file: %s" % file_name)

    def load_params(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
        print('Weights loaded from file %s'%file_name)

    def get_params(self):
        return [param for param in self.net.parameters(True)]

    def set_params(self, input_params):
        for idx, param in enumerate(self.net.parameters(True)):
            param.data = nn.parameter.Parameter(self._to_tensor(input_params[idx].T))
        
    def reset(self):
        pass

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype('float32')).to(DEVICE)

    def _to_numpy(self, x):
        return x.detach().to('cpu').numpy()

    def _get_next_batch(self, database, i):
        batch = database[i*self.batch_size: (i+1)*self.batch_size]
        batch_x = np.array([_[0] for _ in batch])
        batch_y = np.array([_[1] for _ in batch])
        new_shape_ip = [self.batch_size] + [self.output_size]
        new_shape_op = [self.batch_size] + [self.output_size]
        batch_x = batch_x.reshape(new_shape_ip)  
        batch_y = batch_y.reshape(new_shape_op)
        return batch_x, batch_y
    
    def _single_pass(self, x, y):
        out = self.net(x)
        self.optimizer.zero_grad()
        loss = self.loss_function(out, y)
        loss.backward()
        self.optimizer.step()
        return loss.data.item()