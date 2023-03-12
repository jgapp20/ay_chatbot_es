# Este modelo utiliza una red neuronal con tres capas totalmente conectadas. La entrada tiene una forma de 768 (la dimensión del embedding de BERT) y la salida es un solo número que representa la predicción del modelo.
# Se utiliza la función de activación ReLU para las dos primeras capas y una función sigmoidal para la capa de salida, lo que garantiza que la salida del modelo esté entre 0 y 1.
# Se utiliza el optimizador SGD (Gradiente Descendente Estocástico) con una tasa de aprendizaje de 0.01 y la función de pérdida MSE (Mean Squared Error) para entrenar el modelo.

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()
