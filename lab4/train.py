import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0):
        super(MLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        features_2d = self.encoder(x)
        output = self.decoder(features_2d)
        return output, features_2d

input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1
learning_rate = 0.001
num_epochs = 100000

model = MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)
model.load_state_dict(torch.load('mlp_model.pth', map_location=device))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)  

train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model = None

from tqdm import tqdm

# for epoch in tqdm(range(num_epochs)):
#     outputs, _ = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     train_losses.append(loss.item())
    
#     with torch.no_grad():
#         test_outputs, _ = model(X_test_tensor)
#         test_loss = criterion(test_outputs, y_test_tensor)
#         test_losses.append(test_loss.item())
        
#         if test_loss.item() < best_test_loss:
#             best_test_loss = test_loss.item()
#             best_model = model.state_dict().copy()
    
#     if (epoch + 1) % 10000 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# model.load_state_dict(best_model)


with torch.no_grad():
    test_outputs, test_features_2d = model(X_test_tensor)
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

    test_outputs, test_features_2d = model(X_test_tensor)
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
    test_outputs, _ = model(X_test_tensor)
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