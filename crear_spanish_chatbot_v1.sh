#!/bin/bash

# Crear un directorio para el modelo
mkdir spanish_chatbot

# Crear un archivo de configuración para el modelo
echo '{ "input_shape": [1, 1024], "output_shape": [1, 1024], "batch_size": 1 }' > spanish_chatbot/config.json

# Entrenar el modelo de chatbot
python3 -m pip install transformers
python3 -m pip install torch

python3 << END
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

# Cargar el modelo y el tokenizer
model_name = "spanish_urlp"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preparar los datos de entrenamiento
train_texts = [
    "Hola, ¿cómo estás?",
    "¿Qué planes tienes para hoy?",
    "¿Qué opinas sobre la tecnología?",
    "¿Cuál es tu libro favorito?",
    "¿Cómo te llamas?",
    "¿Qué música te gusta?"
]
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")

# Entrenar el modelo
training_args = TrainingArguments(
    output_dir='./spanish_chatbot',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    save_steps=50,
    learning_rate=1e-4,
    seed=42,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings["input_ids"],
)

trainer.train()

# Guardar el modelo
model.save_pretrained('./spanish_chatbot')
tokenizer.save_pretrained('./spanish_chatbot')
END

# Empaquetar el modelo
torch-model-archiver --model-name spanish_chatbot --version 1.0 --serialized-file spanish_chatbot/pytorch_model.bin --handler transformers --extra-files spanish_chatbot/config.json,spanish_chatbot/tokenizer.json --export-path spanish_chatbot

# Registrar el modelo
torchserve --start && \
curl -X POST "http://localhost:8081/models?model_name=spanish_chatbot&url=spanish_chatbot.mar&batch_size=1&max_batch_delay=5000&initial_workers=1&synchronous=true"

# Iniciar el servidor
torchserve --start --model-store model_store

# Ejemplo de uso
curl http://localhost:8080/predictions/spanish_chatbot -T request.json
