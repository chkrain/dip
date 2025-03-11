import torch.nn as nn

class EvolutionNN(nn.Module):
    def __init__(self):
        super(EvolutionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 32),    # Вход: [ресурсы, инструменты, дома, территория, текущий интеллект]
            nn.ReLU(),
            nn.Linear(32, 1),    # Выход: вероятность повышения интеллекта
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)