# evolution_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EvolutionNN(nn.Module):
    def __init__(self):
        super(EvolutionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),  # Увеличили число входных параметров
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Функция для оценки эволюции

def evaluate_evolution(resources, tools, houses, territory, intelligence, age):
    model = EvolutionNN()
    model.load_state_dict(torch.load("evolution_model.pth"))
    model.eval()
    
    input_data = torch.tensor([[resources, tools, houses, territory, intelligence, age]], dtype=torch.float32)
    probability = model(input_data).item()
    
    if probability > 0.51:
        return "Эволюция произошла!"
    else:
        return "Эволюция не произошла."

# Улучшение обучения
def train_evolution_model():
    model = EvolutionNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Пример данных для обучения
    train_X = torch.tensor([
        [150, 3, 2, 30, 5.0, 10],  # Успешная эволюция
        [80, 2, 1, 20, 4.0, 2],    # Неудачная эволюция
        [300, 5, 3, 50, 8.0, 20],  # Успешная эволюция
        [50, 1, 0, 10, 3.0, 1],     # Неудачная эволюция
    ], dtype=torch.float32)

    train_Y = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)
    
    for epoch in range(200000):  # Увеличили число эпох
        optimizer.zero_grad()
        predictions = model(train_X)
        loss = nn.functional.binary_cross_entropy(predictions, train_Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.10f}")
    
    torch.save(model.state_dict(), "evolution_model.pth")
    print("Обучение завершено, модель сохранена!")
    
if __name__ == "__main__":
    train_evolution_model()
