"""
This script fine-tunes the mREBEL model using LoRA (Low-Rank Adaptation).
It prepares the dataset, applies the PEFT (Parameter Efficient Fine-Tuning) approach with LoRA,
and trains the model on the provided triplet data. The fine-tuned model and tokenizer are saved 
after training.

Usage:
Run this script to fine-tune the mREBEL model with LoRA using the triplet data and save the trained model.
"""

import sys
sys.path.insert(0, '../pipeline')
from params import rdf_model, tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import json
from peft import get_peft_model, LoraConfig

# Function to preprocess the data
def preprocess_data(example):
    """
    Preprocesses each example for model training by tokenizing the text 
    and formatting the triplet data with their types for the model.
    
    Args:
        example (dict): The input data example containing 'text' and 'triplets'.
        
    Returns:
        dict: Tokenized inputs with the 'labels' formatted as triplets.
    """
    # Tokenize the text
    inputs = tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)
    
    # Extract the triplets from the example
    triplets = example['triplets']
    
    # Format the triplets by adding head_type and tail_type to each triplet
    triplet_text = " ".join([
        f"{t['head']} ({t['head_type']}) {t['type']} {t['tail']} ({t['tail_type']})"
        for t in triplets
    ])
    
    # Add the formatted triplet text as 'labels' for training
    inputs['labels'] = tokenizer(triplet_text, padding='max_length', truncation=True, max_length=128)['input_ids']
    
    return inputs

if __name__ == "__main__":
    # Load the base mREBEL model and tokenizer
    model = rdf_model

    # Apply QLoRA with the specified LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrix
        lora_alpha=32,  # Scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Specify which model modules to adapt with LoRA
    )

    # Apply PEFT with LoRA to the model
    model = get_peft_model(model, lora_config)

    # Load the training data (triplets) from a JSON file
    with open('data/mrebel_training_data.json', 'r') as f:
        data = json.load(f)

    # Convert the data into Hugging Face Dataset format
    dataset = Dataset.from_dict({
        'text': [item['text'] for item in data],
        'triplets': [item['triplets'] for item in data]
    })

    # Preprocess the data (tokenization and triplet formatting)
    dataset = dataset.map(preprocess_data, remove_columns=['triplets'])

    # Split the dataset into training and testing sets (80% train, 20% test)
    split_datasets = dataset.train_test_split(test_size=0.2)

    # Extract the training and testing datasets
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

    # Set up the training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir='./results',  
        num_train_epochs=10,  
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,  
        warmup_steps=500,  
        weight_decay=0.01,  # Weight decay for regularization
        logging_dir='./logs',  
        logging_steps=10,  
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        fp16=True,  # Enable mixed precision training for faster training (gpu is needed)
    )

    # Initialize the Trainer with the model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model_path = "../../models/finetuned_mrebel"
    model.save_pretrained(model_path)

    # Save the tokenizer
    tokenizer.save_pretrained(model_path)

    print(f"Model saved in the directory: {model_path}")
