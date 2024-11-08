from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# Charger mREBEL et les données
model = AutoModelForSeq2SeqLM.from_pretrained("babelscape/rebel-large")
dataset = load_dataset("votre_dataset_multilingue")

# Préparer les arguments d’entraînement
training_args = TrainingArguments(
    output_dir="./mrebel-finetuned",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Configurer le Trainer pour l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Fine-tuning
trainer.train()
