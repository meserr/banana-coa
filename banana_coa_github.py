import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

class ResizeWithPadding:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        img = F.resize(img, self.size)
        delta_width = self.size[0] - img.size[0]
        delta_height = self.size[1] - img.size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        return F.pad(img, padding, fill=self.fill)

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))
        x = self.dropout(x)
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CheetahOptimizer:
    def __init__(self, population_size, max_iter, models, val_loader, device):
        self.population_size = population_size
        self.max_iter = max_iter
        self.models = models
        self.val_loader = val_loader
        self.device = device
        self.D = len(models)
        self.T = 60 * int(np.ceil(self.D / 10))
        self.population = self._initialize_population()
        self.fitness = np.zeros(population_size)
        self.home_positions = self.population.copy()
        self.leader_position = None
        self.leader_fitness = float('-inf')
        self.prey_position = None
        self.prey_fitness = float('-inf')
        self.t = 0
        self.leader_unchanged_count = 0
        self.convergence_history = []
        
    def _initialize_population(self):
        population = np.random.uniform(0, 1, (self.population_size, self.D))
        population = population / population.sum(axis=1, keepdims=True)
        return population
    
    def _normalize_weights(self, weights):
        weights = np.maximum(weights, 1e-6)
        return weights / weights.sum()
    
    def _calculate_r_hat_inverse(self):
        return np.random.randn(self.D)
    
    def _calculate_alpha(self, t, is_leader=False, cheetah_pos=None, neighbor_pos=None):
        if is_leader:
            return 0.001 * (t / max(self.T, 1)) * 1.0
        else:
            if neighbor_pos is not None:
                return np.abs(neighbor_pos - cheetah_pos)
            else:
                return 0.001 * (t / max(self.T, 1)) * np.ones(self.D)
    
    def _calculate_r_check(self):
        r = np.random.randn(self.D)
        r_check = np.power(np.abs(r) + 1e-10, np.exp(r / 2)) * np.sin(2 * np.pi * r)
        return r_check
    
    def _calculate_beta(self, cheetah_pos, neighbor_pos):
        return neighbor_pos - cheetah_pos
    
    def _calculate_H(self, t, T):
        r1 = np.random.random()
        H = np.exp(2 * (1 - t / max(T, 1))) * (2 * r1 - 1)
        return H
    
    def evaluate_ensemble(self, weights):
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                ensemble_output = torch.zeros((inputs.size(0), 2)).to(self.device)
                
                for i, model in enumerate(self.models):
                    model.eval()
                    output = model(inputs)
                    ensemble_output += weights[i] * output
                
                _, preds = ensemble_output.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return f1_score(all_labels, all_preds, average='weighted')
    
    def _search_strategy(self, cheetah_pos, neighbor_pos, is_leader=False):
        r_hat_inv = self._calculate_r_hat_inverse()
        alpha = self._calculate_alpha(self.t, is_leader, cheetah_pos, neighbor_pos)
        new_pos = cheetah_pos + r_hat_inv * alpha
        return self._normalize_weights(new_pos)
    
    def _sit_and_wait_strategy(self, cheetah_pos):
        return cheetah_pos.copy()
    
    def _attack_strategy(self, cheetah_pos, neighbor_pos):
        r_check = self._calculate_r_check()
        beta = self._calculate_beta(cheetah_pos, neighbor_pos)
        new_pos = self.prey_position + r_check * beta
        return self._normalize_weights(new_pos)
    
    def _leave_prey_and_go_home(self, idx):
        if np.random.random() < 0.5:
            return self._normalize_weights(self.home_positions[idx])
        else:
            return self._normalize_weights(self.prey_position)
    
    def optimize(self):
        for i in range(self.population_size):
            self.fitness[i] = self.evaluate_ensemble(self.population[i])
        
        best_idx = np.argmax(self.fitness)
        self.leader_position = self.population[best_idx].copy()
        self.leader_fitness = self.fitness[best_idx]
        self.prey_position = self.leader_position.copy()
        self.prey_fitness = self.leader_fitness
        
        print(f"COA Initial Best Fitness: {self.prey_fitness:.4f}")
        
        self.t = 0
        it = 1
        prev_leader_fitness = self.leader_fitness
        
        while it <= self.max_iter:
            m = np.random.randint(2, self.population_size + 1)
            selected_indices = np.random.choice(self.population_size, m, replace=False)
            
            for idx in selected_indices:
                neighbor_idx = np.random.choice([j for j in range(self.population_size) if j != idx])
                neighbor_pos = self.population[neighbor_idx]
                
                is_leader = (idx == np.argmax(self.fitness))
                new_pos = np.zeros(self.D)
                
                for j in range(self.D):
                    r2 = np.random.random()
                    r3 = np.random.random()
                    
                    if r2 <= r3:
                        r4 = np.random.uniform(0, 3)
                        H = self._calculate_H(self.t, self.T)
                        
                        if H >= r4:
                            r_check = np.power(np.abs(np.random.randn()) + 1e-10, 
                                             np.exp(np.random.randn() / 2)) * np.sin(2 * np.pi * np.random.randn())
                            beta = neighbor_pos[j] - self.population[idx][j]
                            new_pos[j] = self.prey_position[j] + r_check * beta
                        else:
                            r_hat_inv = np.random.randn()
                            if is_leader:
                                alpha = 0.001 * (self.t / max(self.T, 1))
                            else:
                                alpha = np.abs(neighbor_pos[j] - self.population[idx][j])
                            new_pos[j] = self.population[idx][j] + r_hat_inv * alpha
                    else:
                        new_pos[j] = self.population[idx][j]
                
                new_pos = self._normalize_weights(new_pos)
                new_fitness = self.evaluate_ensemble(new_pos)
                
                if new_fitness > self.fitness[idx]:
                    self.population[idx] = new_pos
                    self.fitness[idx] = new_fitness
                    
                    if new_fitness > self.leader_fitness:
                        self.leader_position = new_pos.copy()
                        self.leader_fitness = new_fitness
            
            self.t += 1
            
            if self.t > np.random.random() * self.T:
                if self.leader_fitness <= prev_leader_fitness:
                    self.leader_unchanged_count += 1
                    if self.leader_unchanged_count > 3:
                        for idx in selected_indices:
                            new_pos = self._leave_prey_and_go_home(idx)
                            new_fitness = self.evaluate_ensemble(new_pos)
                            if new_fitness > self.fitness[idx]:
                                self.population[idx] = new_pos
                                self.fitness[idx] = new_fitness
                        self.t = 0
                        self.leader_unchanged_count = 0
                else:
                    self.leader_unchanged_count = 0
            
            prev_leader_fitness = self.leader_fitness
            
            if self.leader_fitness > self.prey_fitness:
                self.prey_position = self.leader_position.copy()
                self.prey_fitness = self.leader_fitness
            
            self.convergence_history.append(self.prey_fitness)
            
            if it % 10 == 0:
                print(f"COA Iteration {it}/{self.max_iter}, Best Fitness: {self.prey_fitness:.4f}")
            
            it += 1
        
        print(f"COA Final Best Fitness: {self.prey_fitness:.4f}")
        print(f"COA Best Weights: {self.prey_position}")
        
        return self.prey_position

def train_model(model, train_loader, val_loader, device, epochs=YOUR_EPOCHS, patience=YOUR_PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=YOUR_LEARNING_RATE)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.size(0) != labels.size(0):
                    continue
                    
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return history

def get_transform():
    return transforms.Compose([
        ResizeWithPadding((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_dataset(data_path):
    return torchvision.datasets.ImageFolder(data_path, transform=transform)

def train_and_evaluate(dataset):
    indices = torch.randperm(len(dataset))[:YOUR_DATASET_SIZE]
    limited_dataset = Subset(dataset, indices)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset size: {len(limited_dataset)}")
    
    sample_input, _ = limited_dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    
    models = [
        CNN1().to(device),
        CNN2().to(device),
        CNN3().to(device)
    ]
    
    kf = KFold(n_splits=YOUR_SPLIT_SIZE, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(limited_dataset)))):
        print(f"\nFold {fold + 1}")
        print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
        
        train_subset = Subset(limited_dataset, train_idx)
        val_subset = Subset(limited_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=YOUR_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=YOUR_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)
        
        for model_idx, model in enumerate(models):
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            train_model(model, train_loader, val_loader, device)
        
        optimizer = CheetahOptimizer(
            population_size=YOUR_POPULATION_SIZE,
            max_iter=YOUR_MAX_ITERATIONS,
            models=models, 
            val_loader=val_loader, 
            device=device
        )
        best_weights = optimizer.optimize()

if __name__ == "__main__":
    YOUR_EPOCHS = 10
    YOUR_PATIENCE = 5
    YOUR_LEARNING_RATE = 0.001
    YOUR_SPLIT_SIZE = 5
    YOUR_DATASET_SIZE = 1000
    YOUR_BATCH_SIZE = 32
    YOUR_POPULATION_SIZE = 10
    YOUR_MAX_ITERATIONS = 100
    YOUR_DATA_PATH = "path/to/your/data/"
    
    dataset = load_dataset(YOUR_DATA_PATH)
    train_and_evaluate(dataset)
