import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import DEVICE

class _SupervisedCnn(nn.Module):
    def __init__(self, input_size, output_size, patch_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)

        # CNN Parameters
        IN_CH = 3
        KERNEL_SIZE = 3
        STRIDE = 2
        PADDING = 2
        CH1 = 6
        CH2 = 10
        INTERMEDIATE_OUTPUT_SHAPE = 160
        CNN_OUT_SIZE = 50

        self.cnn1 = nn.Conv2d(IN_CH, CH1, KERNEL_SIZE, STRIDE, PADDING)
        self.cnn2 = nn.Conv2d(CH1, CH2, KERNEL_SIZE, STRIDE, PADDING)
        self.cnn_linear = nn.Linear(INTERMEDIATE_OUTPUT_SHAPE, CNN_OUT_SIZE)

        self.final_linear = nn.Linear(50 + CNN_OUT_SIZE, output_size)
        
    def forward(self, x_linear, patch):
        x_linear = F.relu(self.fc1(x_linear))
        x_linear = F.relu(self.fc2(x_linear))
        x_cnn = F.relu(self.cnn1(patch))
        x_cnn = F.relu(self.cnn2(x_cnn))
        x_cnn = x_cnn.view(x_cnn.shape[0], -1)
        x_cnn = F.relu(self.cnn_linear(x_cnn))
        out = torch.cat([x_linear, x_cnn], 1)
        out = self.final_linear(out)
        return out

class SupervisedCnn():
    def __init__(self, params):
        DEFAULT_PATCH_SIZE = 5

        self.use_image_patch = True
        self.initialized=False
        self.output_size = params['output_size']
        self.input_size = params['input_size']
        self.learning_rate = params['learning_rate']
        self.batch_size =  params['batch_size']
        self.training_epochs = params['training_epochs']
        self.display_step = params['display_step']
        self.seed_val = params['seed_val']
        self.patch_size = params['patch_size'] if 'patch_size' in params else DEFAULT_PATCH_SIZE

    def initialize(self):
        """
        Initialize network with weights dependent on seed
        """
        if not self.initialized:
            torch.random.manual_seed(self.seed_val)
            np.random.seed(self.seed_val)
            random.seed(self.seed_val)
            self.net = _SupervisedCnn(self.input_size, self.output_size, self.patch_size).to(DEVICE)
            self.initialized = True
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.learning_rate)
            self.loss_function = nn.MSELoss()
    def create_network(self):
        """
        Kept for legacy reasons. initialize() should create the network entirely
        """
        pass

    def train(self, database):
        random.seed(0)
        avg_loss = 0
        epoch_loss_hist = []
        for epoch_idx in range(self.training_epochs):
            random.shuffle(database)
            total_batch = int(len(database)/self.batch_size)
            avg_loss = 0
            for idx in range(total_batch):
                batch_x, batch_patch, batch_y = self._get_next_batch(database, idx)
                loss = self._single_pass(batch_x, batch_patch, batch_y)
                # TODO: Fix this weird loss statistics
                avg_loss += loss / total_batch
            if epoch_idx % self.display_step == 0:
                print ("epoch:", '%04d' % (epoch_idx + 1), "cost=", \
              "{:.9f}".format(np.sqrt(avg_loss)))
            epoch_loss_hist.append(avg_loss)
        print("Optimization Finished")
        return np.sqrt(avg_loss), epoch_loss_hist

    def get_loss(self, features, label):
        pass

    def get_heuristic(self, features):
        """
        Returns a single heuristic value corresponding to a single input
        """
        patch = features[1]
        features = features[0]
        x = torch.unsqueeze(self._to_tensor(features), 0)
        patch = torch.unsqueeze(self._to_tensor(patch), 0)
        with torch.no_grad():
            out = self.net(x, patch)
        return self._to_numpy(out).item()

    def save_params(self, file_name):
        """
        Will save file as "file_name.weights"
        """
        weights_file_name = "{}.weights".format(file_name)
        torch.save(self.net.state_dict(), weights_file_name)
        print("Model saved in file: %s" % weights_file_name)

    def load_params(self, file_name):
        """
        Will load file "file_name.weights"
        """
        weights_file_name = "{}.weights".format(file_name)
        self.net.load_state_dict(torch.load(weights_file_name))
        print('Weights loaded from file %s'%weights_file_name)

    def get_params(self):
        return [param for param in self.net.parameters(True)]

    def set_params(self, input_params):
        """
        Input should be a list of ndarray with elements
        0: W1 shape (input, l1)
        1: b1 shape (l1,)
        2: W2 shape(l1, l2)
        3: b2 shape (l2)
        4: W3 shape (l2, output)
        5: b3 shape (output,)
        """
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
        batch_x = np.array([_[0][0] for _ in batch])
        batch_patch = np.array([_[0][1] for _ in batch])
        batch_y = np.array([_[1] for _ in batch])
        new_shape_ip = [self.batch_size] + [self.input_size]
        new_shape_op = [self.batch_size] + [self.output_size]
        batch_x = batch_x.reshape(new_shape_ip)  
        batch_y = batch_y.reshape(new_shape_op)
        return batch_x, batch_patch, batch_y
    
    def _single_pass(self, x, patch, y):
        out = self.net(self._to_tensor(x), self._to_tensor(patch))
        self.optimizer.zero_grad()
        loss = self.loss_function(out, self._to_tensor(y))
        loss.backward()
        self.optimizer.step()
        return loss.data.item()