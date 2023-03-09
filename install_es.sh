#!/bin/bash

# Instalar PyTorch y PyTorch serve
pip3 install torch torchvision torchaudio
pip3 install torchserve torch-model-archiver

# Instalar modelo de chatbot en español
pip3 install spanish_chatbot==0.0.2



#!/bin/bash

# Instalar PyTorch
pip3 install torch

# Instalar PyTorch Serve
pip3 install torchserve torch-model-archiver

# matplotlib
sudo apt-get install python3-matplotlib
python -m pip install -U pip
python -m pip install -U matplotlib

# Clonar el modelo de Spanish_chatbot
git clone https://github.com/jgapp20/ay_chatbot_es/spanish-chatbot.git

# Entrar en la carpeta del modelo
cd spanish-chatbot

# Descargar los pesos del modelo
wget https://github.com/jgapp20/ay_chatbot_es/releases/download/v1.0.0/best_model.pt

# Crear un archivo de configuración para el modelo
echo '{ "input_shape": [1,512], "output_shape": [1,768], "batch_size": 1 }' > config.json

# Empaquetar el modelo
torch-model-archiver --model-name spanish_chatbot --version 1.0 --serialized-file best_model.pt --handler chatbot_handler.py --extra-files config.json

# Registrar el modelo
torchserve --start && \
curl -X POST "http://localhost:8081/models?model_name=spanish_chatbot&url=spanish_chatbot.mar&batch_size=1&max_batch_delay=5000&initial_workers=1&synchronous=true"

# Iniciar el servidor
torchserve --start --model-store model_store

# Ejemplo de uso
curl http://localhost:8080/predictions/spanish_chatbot -T request.json
