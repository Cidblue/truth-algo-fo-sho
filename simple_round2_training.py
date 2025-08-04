#!/usr/bin/env python3
"""
Simplified Round 2 Training - 28 Categories
Reliable training script without complex features that might hang
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load and prepare training data"""
    print("📊 Loading 28-category dataset...")
    
    # Load data
    df = pd.read_csv("data/complete_28_category_dataset.csv")
    labeled_df = df[df['label'].notna() & (df['label'] != '')]
    
    # Load mapping
    with open('official_28_category_mapping.json', 'r') as f:
        mapping_data = json.load(f)
    
    label_to_id = mapping_data['label_to_id']
    id_to_label = mapping_data['id_to_label']
    
    # Prepare data
    texts = []
    labels = []
    
    for _, row in labeled_df.iterrows():
        text = row['text']
        label_str = row['label']
        
        # Take first label if multiple
        if ',' in label_str:
            primary_label = label_str.split(',')[0].strip()
        else:
            primary_label = label_str.strip()
        
        if primary_label in label_to_id:
            texts.append(text)
            labels.append(label_to_id[primary_label])
    
    print(f"✅ Loaded {len(texts)} training examples")
    print(f"✅ Using {len(label_to_id)} categories")
    
    return texts, labels, label_to_id, id_to_label

def train_model():
    """Train the model"""
    print("🚀 Starting Round 2 Training...")
    
    # Load data
    texts, labels, label_to_id, id_to_label = load_data()
    
    # Split data (no stratification to avoid issues)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Initialize model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id)
    )
    
    print(f"📋 Model: {model_name}")
    
    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train, tokenizer)
    val_dataset = SimpleDataset(X_val, y_val, tokenizer)
    
    # Simple training arguments
    training_args = TrainingArguments(
        output_dir="models/round2-simple",
        num_train_epochs=3,  # Reduced for faster training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to=[],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("🏋️ Training started...")
    start_time = datetime.now()
    
    # Train
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"✅ Training completed in {duration}")
    
    # Save model
    model.save_pretrained("models/round2-simple")
    tokenizer.save_pretrained("models/round2-simple")
    
    # Save mapping
    with open("models/round2-simple/label_map.json", "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "num_labels": len(label_to_id)
        }, f, indent=2)
    
    print("💾 Model saved to models/round2-simple/")
    
    # Quick evaluation
    print("📊 Evaluating...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"🎯 Validation Accuracy: {accuracy:.3f}")
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "training_duration": str(duration),
        "model_name": model_name,
        "num_labels": len(label_to_id),
        "training_examples": len(X_train),
        "validation_examples": len(X_val),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("round2_simple_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("📊 Results saved to round2_simple_results.json")
    
    return results

def main():
    try:
        print("🚀 ROUND 2 SIMPLE TRAINING - 28 CATEGORIES")
        print("="*50)
        
        # Check prerequisites
        if not Path("data/complete_28_category_dataset.csv").exists():
            print("❌ Complete dataset not found!")
            return 1
        
        if not Path("official_28_category_mapping.json").exists():
            print("❌ Category mapping not found!")
            return 1
        
        # Train
        results = train_model()
        
        print(f"\n🎉 SUCCESS!")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Duration: {results['training_duration']}")
        print(f"Model: models/round2-simple/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
