# Importer les bibliothèques nécessaires
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

# Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prétraiter les données
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Prétraitement des données 
tokenized_imdb = data.map(preprocess_function, batched=True)

# Configuration du DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Définition des labels
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# Chargement de la métrique d'exactitude
accuracy = evaluate.load("accuracy")

# Fonction de calcul des métriques
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Chargement du modèle
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Configuration des arguments d'entraînement
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,  # Sauvegarde toutes les 1000 étapes
    eval_steps=1000,  # Évaluation toutes les 1000 étapes
    load_best_model_at_end=True,
    save_total_limit=3,  # Conserver uniquement les 3 derniers checkpoints
)

# Initialisation du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Entraînement du modèle
trainer.train()
