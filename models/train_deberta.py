"""
Script to fine-tune a DeBERTa model for outpoint/pluspoint classification.
"""
import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch
from sklearn.model_selection import train_test_split
import argparse

def prepare_dataset(csv_path, text_column="text", label_column="label"):
    """Prepare dataset from CSV file."""
    # Load data
    df = pd.read_csv(csv_path)
    
    # Get unique labels and create mappings
    labels = sorted(df[label_column].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    # Convert labels to IDs
    df["label_id"] = df[label_column].map(label2id)
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_column])
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset, label2id, id2label

def tokenize_function(examples, tokenizer, text_column="text"):
    """Tokenize examples."""
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=512)

def main(args):
    # Prepare dataset
    train_dataset, val_dataset, label2id, id2label = prepare_dataset(
        args.data_path, args.text_column, args.label_column
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.text_column),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.text_column),
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")
    print(f"Label mapping: {id2label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="models/deberta-lora", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    
    args = parser.parse_args()
    main(args)