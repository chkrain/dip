# train_evolution.py
import torch
import torch.nn as nn
import torch.optim as optim

class EvolutionNN(nn.Module):
    def __init__(self):
        super(EvolutionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Инициализация модели и оптимизатора
model = EvolutionNN()
optimizer = optim.Adam(model.parameters())

# Генерация тренировочных данных
train_X = torch.tensor([
    [150, 3, 2, 30, 5.0],  # Пример успешной эволюции
    [80,  2, 1, 20, 4.0],   # Пример неудачи
    [300, 5, 3, 50, 8.0]    # Пример успеха
], dtype=torch.float32)

train_Y = torch.tensor([[1], [0], [1]], dtype=torch.float32)

# Процесс обучения
for epoch in range(10000):
    optimizer.zero_grad()
    predictions = model(train_X)
    loss = nn.functional.binary_cross_entropy(predictions, train_Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Сохранение модели
torch.save(model.state_dict(), "evolution_model.pth")
print("Модель успешно обучена и сохранена!")