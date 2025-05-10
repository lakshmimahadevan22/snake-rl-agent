import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime

class DQNetwork(nn.Module):

    
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.norm = nn.LayerNorm(hidden_layers[0])
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.leaky_relu(self.layers[0](x))
        x = self.norm(x)
        
        for i in range(1, len(self.layers)):
            x = F.leaky_relu(self.layers[i](x))
            
        x = self.output(x)
        return x
    
    def save(self, file_name=None):
        model_folder_path = './trained_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
    
        if file_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f'model_{timestamp}.pth'
            

        file_path = os.path.join(model_folder_path, file_name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_architecture': {
                'input_size': self.layers[0].in_features,
                'hidden_layers': [layer.out_features for layer in self.layers],
                'output_size': self.output.out_features
            }
        }, file_path)
        
        print(f'Model saved to {file_path}')
        
    @classmethod
    def load(cls, file_path):
        """Load model from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")
            
        checkpoint = torch.load(file_path)
        architecture = checkpoint['model_architecture']
    
        model = cls(
            input_size=architecture['input_size'],
            hidden_layers=architecture['hidden_layers'],
            output_size=architecture['output_size']
        )
        
    
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() 
        
        return model


class DQNTrainer:    
    def __init__(self, model, lr, discount, target_update=10):
        self.lr = lr
        self.discount = discount
        self.model = model
    
        self.target_model = DQNetwork(
            self.model.layers[0].in_features,
            [layer.out_features for layer in self.model.layers],
            self.model.output.out_features
        )
        self.update_target_network()
        self.target_update_counter = 0
        self.target_update_freq = target_update
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def optimize(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        current_q = self.model(state)
        
        target_q = current_q.clone()
        
        batch_size = len(done)
        for idx in range(batch_size):
            if done[idx]:
                new_q = reward[idx]
            else:
                next_q = self.target_model(next_state[idx])
                new_q = reward[idx] + self.discount * torch.max(next_q)
            target_q[idx][torch.argmax(action[idx]).item()] = new_q
        
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.update_target_network()
            self.target_update_counter = 0
            
        return loss.item()