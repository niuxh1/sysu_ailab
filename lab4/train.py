import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error




data = pd.read_csv('MLP_data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0):
        # 初始化权重和偏置，使用Xavier/Glorot初始化
        self.W1 = torch.randn(input_size, hidden_size1) * np.sqrt(2.0 / (input_size + hidden_size1))
        self.b1 = torch.zeros(hidden_size1)
        self.W2 = torch.randn(hidden_size1, 2) * np.sqrt(2.0 / (hidden_size1 + 2))
        self.b2 = torch.zeros(2)
        self.W3 = torch.randn(2, hidden_size2) * np.sqrt(2.0 / (2 + hidden_size2))
        self.b3 = torch.zeros(hidden_size2)
        self.W4 = torch.randn(hidden_size2, output_size) * np.sqrt(2.0 / (hidden_size2 + output_size))
        self.b4 = torch.zeros(output_size)
        
        self.dropout_rate = dropout_rate
        self.parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        self.cache = {}
    
    def relu(self, x):
        return torch.max(torch.zeros_like(x), x)
    
    def dropout(self, x, training=True):
        if not training or self.dropout_rate == 0:
            return x
        mask = (torch.rand_like(x) > self.dropout_rate).float()
        return x * mask / (1 - self.dropout_rate)
    
    def forward(self, x, training=True):
        # Encoder
        z1 = torch.mm(x, self.W1) + self.b1
        a1 = self.relu(z1)
        a1_dropout = self.dropout(a1, training)
        
        z2 = torch.mm(a1_dropout, self.W2) + self.b2
        features_2d = z2  # 2D特征
        
        # Decoder
        z3 = torch.mm(features_2d, self.W3) + self.b3
        a3 = self.relu(z3)
        a3_dropout = self.dropout(a3, training)
        
        z4 = torch.mm(a3_dropout, self.W4) + self.b4
        
        # 保存用于反向传播的中间值
        if training:
            self.cache = {
                'x': x,
                'z1': z1, 'a1': a1, 'a1_dropout': a1_dropout,
                'z2': z2, 'features_2d': features_2d,
                'z3': z3, 'a3': a3, 'a3_dropout': a3_dropout,
                'z4': z4
            }
        
        return z4, features_2d
    
    def backward(self, outputs, targets):
        batch_size = outputs.shape[0]
        
        # 输出层误差
        dz4 = 2.0 * (outputs - targets) / batch_size  # MSE的导数
        
        # 输出层权重和偏置的梯度
        dW4 = torch.mm(self.cache['a3_dropout'].t(), dz4)
        db4 = torch.sum(dz4, dim=0)
        
        # 反向传播到隐藏层3
        da3_dropout = torch.mm(dz4, self.W4.t())
        da3 = da3_dropout / (1 - self.dropout_rate) if self.dropout_rate > 0 else da3_dropout
        dz3 = da3 * (self.cache['z3'] > 0).float()  # ReLU导数
        
        # 隐藏层3权重和偏置的梯度
        dW3 = torch.mm(self.cache['features_2d'].t(), dz3)
        db3 = torch.sum(dz3, dim=0)
        
        # 反向传播到2D特征层
        dfeatures = torch.mm(dz3, self.W3.t())
        
        # 2D特征层权重和偏置的梯度
        dW2 = torch.mm(self.cache['a1_dropout'].t(), dfeatures)
        db2 = torch.sum(dfeatures, dim=0)
        
        # 反向传播到隐藏层1
        da1_dropout = torch.mm(dfeatures, self.W2.t())
        da1 = da1_dropout / (1 - self.dropout_rate) if self.dropout_rate > 0 else da1_dropout
        dz1 = da1 * (self.cache['z1'] > 0).float()  # ReLU导数
        
        # 隐藏层1权重和偏置的梯度
        dW1 = torch.mm(self.cache['x'].t(), dz1)
        db1 = torch.sum(dz1, dim=0)
        
        return [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
    
    def update_parameters(self, gradients, learning_rate, weight_decay=0):
        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            # 应用L2正则化
            if i % 2 == 0 and weight_decay > 0:  # 只对权重应用正则化，不对偏置应用
                param.data -= learning_rate * (grad + weight_decay * param)
            else:
                param.data -= learning_rate * grad
    
    def state_dict(self):
        return {
            'W1': self.W1.clone(), 'b1': self.b1.clone(),
            'W2': self.W2.clone(), 'b2': self.b2.clone(),
            'W3': self.W3.clone(), 'b3': self.b3.clone(),
            'W4': self.W4.clone(), 'b4': self.b4.clone()
        }
    
    def load_state_dict(self, state_dict):
        self.W1 = state_dict['W1']
        self.b1 = state_dict['b1']
        self.W2 = state_dict['W2']
        self.b2 = state_dict['b2']
        self.W3 = state_dict['W3']
        self.b3 = state_dict['b3']
        self.W4 = state_dict['W4']
        self.b4 = state_dict['b4']
        self.parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]

# 定义MSE损失函数
def mse_loss(outputs, targets):
    return torch.mean((outputs - targets) ** 2)

input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1
learning_rate = 0.001
num_epochs = 100000


model = MLP(input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0)

try:
    saved_state = torch.load('mlp_model.pth')

    if 'W1' in saved_state:
        model.load_state_dict(saved_state)

except:
    print("未找到模型或模型格式不兼容，将使用新初始化的模型")

train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model = None

from tqdm import tqdm

for epoch in tqdm(range(num_epochs)):

    outputs, _ = model.forward(X_train_tensor)
    loss = mse_loss(outputs, y_train_tensor)
    

    gradients = model.backward(outputs, y_train_tensor)
    

    model.update_parameters(gradients, learning_rate, weight_decay=1e-5)
    
    train_losses.append(loss.item())
    

    with torch.no_grad():
        test_outputs, _ = model.forward(X_test_tensor, training=False)
        test_loss = mse_loss(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
        
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_model = model.state_dict()
    
    if (epoch + 1) % 10000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

model.load_state_dict(best_model)


with torch.no_grad():
    test_outputs, test_features_2d = model.forward(X_test_tensor, training=False)  
    test_predictions = test_outputs.cpu().numpy().flatten()
    
    test_predictions_original = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predictions_original))
    print(f'Test RMSE: {test_rmse:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')

plt.figure(figsize=(10, 8))
with torch.no_grad():

    test_outputs, test_features_2d = model.forward(X_test_tensor, training=False)  
    test_predictions = test_outputs.cpu().numpy().flatten()
    test_features_2d = test_features_2d.cpu().numpy()

    test_predictions_original = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    plt.scatter(test_features_2d[:, 0], test_features_2d[:, 1], c=y_test_original, cmap='viridis', 
                alpha=0.8, s=100, label='Actual Prices', edgecolors='k')

    plt.scatter(test_features_2d[:, 0], test_features_2d[:, 1], c=test_predictions_original, 
                cmap='plasma', alpha=0.8, s=50, marker='x', label='Predicted Prices')
    
    cbar = plt.colorbar()
    cbar.set_label('House Price')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Actual vs Predicted House Prices in 2D Feature Space')
    plt.grid(True)
    plt.legend()
    plt.savefig('price_comparison_2d.png')


plt.figure(figsize=(10, 10))
with torch.no_grad():
    test_outputs, _ = model.forward(X_test_tensor, training=False)  
    test_predictions = test_outputs.cpu().numpy().flatten()
 
    test_predictions_original = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    plt.scatter(test_predictions_original, y_test_original, alpha=0.7)

    min_val = min(test_predictions_original.min(), y_test_original.min())
    max_val = max(test_predictions_original.max(), y_test_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    plt.savefig('predicted_vs_actual.png')

torch.save(model.state_dict(), 'mlp_model.pth')