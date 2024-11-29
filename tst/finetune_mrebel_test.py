from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import json
from peft import get_peft_model, LoraConfig

# Charger mREBEL et les données
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", tgt_lang="en_XX")

# Appliquer QLoRA avec la configuration de LoRA
lora_config = LoraConfig(
    r=8,  # Rang de la matrice de Low-Rank
    lora_alpha=32,  # Facteur d'échelle
    lora_dropout=0.1,  # Taux de drop-out pour le Lora
    target_modules=["q_proj", "v_proj"],  # Modules à adapter via LoRA
)

# Appliquer le modèle PEFT avec QLoRA
model = get_peft_model(model, lora_config)

# Fonction pour préparer le dataset
def preprocess_data(example):
    # Tokeniser le texte
    inputs = tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)
    
    # Créer un format de sortie pour les triplets avec types head_type et tail_type
    triplets = example['triplets']
    
    # Pour chaque triplet, on crée une chaîne incluant head_type et tail_type
    triplet_text = " ".join([
        f"{t['head']} ({t['head_type']}) {t['type']} {t['tail']} ({t['tail_type']})"
        for t in triplets
    ])
    
    # Ajouter la sortie dans les labels pour que le modèle l'apprenne
    inputs['labels'] = tokenizer(triplet_text, padding='max_length', truncation=True, max_length=128)['input_ids']
    
    return inputs

# Charger le fichier JSON
with open('mrebel_training_data.json', 'r') as f:
    data = json.load(f)

# Convertir les données en format Dataset de Hugging Face
dataset = Dataset.from_dict({
    'text': [item['text'] for item in data],
    'triplets': [item['triplets'] for item in data]
})

# Prétraiter les données
dataset = dataset.map(preprocess_data, remove_columns=['triplets'])

# Diviser le dataset en train/test
train_dataset = dataset.train_test_split(test_size=0.2)['train']
test_dataset = dataset.train_test_split(test_size=0.2)['test']

# Spécifier les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True,  # Utilisation de la précision flottante 16-bit pour accélérer l'entraînement
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Lancer l'entraînement
trainer.train()
