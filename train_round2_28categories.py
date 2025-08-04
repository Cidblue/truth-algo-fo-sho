#!/usr/bin/env python3
"""
Round 2 DeBERTa Training with Complete 28-Category Dataset
CRITICAL: This script uses the complete 28-category dataset (not the incomplete 17-category one)
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
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

def load_complete_28_category_data():
    """Load the complete 28-category dataset"""
    print("🔍 LOADING COMPLETE 28-CATEGORY DATASET")
    print("="*50)
    
    # Load the complete dataset
    df = pd.read_csv("data/complete_28_category_dataset.csv")
    
    # Filter labeled data
    labeled_df = df[df['label'].notna() & (df['label'] != '')]
    
    print(f"📊 Total statements: {len(df)}")
    print(f"📊 Labeled statements: {len(labeled_df)}")
    
    # Load official mapping
    with open('official_28_category_mapping.json', 'r') as f:
        mapping_data = json.load(f)
    
    label_to_id = mapping_data['label_to_id']
    id_to_label = mapping_data['id_to_label']
    
    print(f"📊 Official categories: {len(label_to_id)}")
    
    # Prepare training data
    texts = []
    labels = []
    
    for _, row in labeled_df.iterrows():
        text = row['text']
        label_str = row['label']
        
        # Handle multiple labels (take first one for now)
        if ',' in label_str:
            primary_label = label_str.split(',')[0].strip()
        else:
            primary_label = label_str.strip()
        
        if primary_label in label_to_id:
            texts.append(text)
            labels.append(label_to_id[primary_label])
        else:
            print(f"⚠️ Unknown label: {primary_label}")
    
    print(f"📊 Training examples: {len(texts)}")
    
    # Validate we have all 28 categories
    unique_labels = set(labels)
    expected_categories = set(range(len(label_to_id)))
    missing_categories = expected_categories - unique_labels
    
    if missing_categories:
        missing_names = [id_to_label[str(cat_id)] for cat_id in missing_categories]
        print(f"⚠️ Missing categories in training data: {missing_names}")
    else:
        print(f"✅ All {len(label_to_id)} categories present in training data!")
    
    return texts, labels, label_to_id, id_to_label

def analyze_data_distribution(labels, id_to_label):
    """Analyze the distribution of training data"""
    print(f"\n📈 DATA DISTRIBUTION ANALYSIS")
    print("="*40)
    
    from collections import Counter
    label_counts = Counter(labels)
    
    print(f"{'Category':<40} {'Count'}")
    print("-" * 50)
    
    outpoint_total = 0
    pluspoint_total = 0
    
    for label_id in sorted(label_counts.keys()):
        category_name = id_to_label[str(label_id)]
        count = label_counts[label_id]
        
        if category_name.endswith('_OUT'):
            outpoint_total += count
        elif category_name.endswith('_PLUS'):
            pluspoint_total += count
        
        print(f"{category_name:<40} {count:>5}")
    
    print("-" * 50)
    print(f"{'OUTPOINTS TOTAL':<40} {outpoint_total:>5}")
    print(f"{'PLUSPOINTS TOTAL':<40} {pluspoint_total:>5}")
    print(f"{'NEUTRAL + OTHER':<40} {sum(label_counts.values()) - outpoint_total - pluspoint_total:>5}")
    
    balance_ratio = pluspoint_total / outpoint_total if outpoint_total > 0 else 0
    print(f"{'BALANCE RATIO (Plus:Out)':<40} {balance_ratio:>5.2f}")

def train_round2_model():
    """Train Round 2 model with complete 28-category dataset"""
    print("🚀 TRAINING ROUND 2 MODEL - 28 CATEGORIES")
    print("="*50)
    
    # Load data
    texts, labels, label_to_id, id_to_label = load_complete_28_category_data()
    
    # Analyze distribution
    analyze_data_distribution(labels, id_to_label)
    
    # Check for categories with too few examples
    from collections import Counter
    label_counts = Counter(labels)
    rare_categories = [label for label, count in label_counts.items() if count < 2]

    if rare_categories:
        rare_names = [id_to_label[str(label)] for label in rare_categories]
        print(f"⚠️ Categories with <2 examples (cannot stratify): {rare_names}")
        print("Using random split instead of stratified split")

        # Use random split instead of stratified
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    else:
        # Use stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
    print(f"\n📊 TRAIN/VALIDATION SPLIT")
    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    
    # Initialize tokenizer and model - using BERT for reliability
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    num_labels = len(label_to_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    print(f"📋 Model: {model_name}")
    print(f"📋 Number of labels: {num_labels}")
    
    # Create datasets
    train_dataset = TruthDataset(X_train, y_train, tokenizer)
    val_dataset = TruthDataset(X_val, y_val, tokenizer)
    
    # Training arguments - optimized for Round 2
    training_args = TrainingArguments(
        output_dir="models/round2-28categories",
        num_train_epochs=8,  # Increased from 6
        per_device_train_batch_size=8,  # Reduced for stability
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="logs/round2-training",
        logging_steps=10,
        eval_strategy="steps",  # Fixed parameter name
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=1e-5,  # Reduced for fine-tuning
        save_total_limit=3,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to=[],  # Disable wandb
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print(f"\n🏋️ STARTING TRAINING")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    
    # Train model
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    training_duration = end_time - start_time
    print(f"✅ Training completed in: {training_duration}")
    
    # Save model and tokenizer
    model.save_pretrained("models/round2-28categories")
    tokenizer.save_pretrained("models/round2-28categories")
    
    # Save label mapping
    with open("models/round2-28categories/label_map.json", "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "num_labels": num_labels
        }, f, indent=2)
    
    print(f"💾 Model saved to: models/round2-28categories/")
    
    # Evaluate model
    print(f"\n📊 EVALUATING ROUND 2 MODEL")
    eval_results = trainer.evaluate()
    
    # Make predictions on validation set
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Classification report
    target_names = [id_to_label[str(i)] for i in range(num_labels)]
    report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
    
    # Save results
    results = {
        "training_duration": str(training_duration),
        "eval_results": eval_results,
        "classification_report": report,
        "model_info": {
            "base_model": model_name,
            "num_labels": num_labels,
            "training_examples": len(X_train),
            "validation_examples": len(X_val),
            "total_categories": len(label_to_id)
        },
        "training_args": {
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"round2_training_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📊 Results saved to: {results_file}")
    
    # Print summary
    accuracy = report['accuracy']
    macro_avg_f1 = report['macro avg']['f1-score']
    
    print(f"\n🎯 ROUND 2 RESULTS SUMMARY")
    print("="*30)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_avg_f1:.3f}")
    print(f"Categories: {num_labels}")
    print(f"Training examples: {len(X_train)}")
    
    return results

def main():
    try:
        print("🚀 ROUND 2 DEBERTA TRAINING - 28 CATEGORIES")
        print("="*60)
        
        # Check if complete dataset exists
        if not Path("data/complete_28_category_dataset.csv").exists():
            print("❌ ERROR: Complete 28-category dataset not found!")
            print("Run: python fix_28_categories.py first")
            return 1
        
        if not Path("official_28_category_mapping.json").exists():
            print("❌ ERROR: Official category mapping not found!")
            print("Run: python fix_28_categories.py first")
            return 1
        
        # Train model
        results = train_round2_model()
        
        print(f"\n🎉 ROUND 2 TRAINING COMPLETE!")
        print(f"🎯 Next: Test the model and validate improvements")
        print(f"📁 Model location: models/round2-28categories/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
