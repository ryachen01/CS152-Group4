import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',  
            max_length=self.max_length,
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


class BertClassifier():
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.num_labels = None  
        self.label_map = None  
        self.model = None 
        print(f"\nUsing model: {model_name}")

    def collate_fn(self, batch):
        return {
            'input_ids': torch.nn.utils.rnn.pad_sequence(
                [item['input_ids'].detach().clone() for item in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ),
            'attention_mask': torch.nn.utils.rnn.pad_sequence(
                [item['attention_mask'].detach().clone() for item in batch],
                batch_first=True,
                padding_value=0
            ),
            'labels': torch.tensor([item['labels'] for item in batch])
        }

    def load_data(self, dataset_path):
        try:
            df = pd.read_csv(dataset_path, names=['text', 'class']).reset_index(drop=True)
            df = df.drop(index=0)

            print(f"Original dataset size: {len(df)} samples")
            print("\nOriginal class distribution:")
            print(df['class'].value_counts())

            unique_labels = sorted(df['class'].unique())
            self.num_labels = len(unique_labels)
            print(f"\nNumber of unique classes: {self.num_labels}")
            print("Classes:", unique_labels)
            
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
            print("\nLabel mapping:", self.label_map)
            
            # Get samples for each class
            class_0_samples = df[df['class'] == self.reverse_label_map[0]]
            class_1_samples = df[df['class'] == self.reverse_label_map[1]]
            
            # Calculate how many samples we need for 90-10 split
            total_samples = min(len(class_0_samples), len(class_1_samples) * 9)  # 9 times more class 0 than class 1
            class_1_count = total_samples // 10
            class_0_count = total_samples - class_1_count
            
            # Sample the required number of examples
            class_0_sampled = class_0_samples.sample(n=class_0_count, random_state=42)
            class_1_sampled = class_1_samples.sample(n=class_1_count, random_state=42)
            
            # Combine the samples
            df = pd.concat([class_0_sampled, class_1_sampled])
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print(f"\nNew dataset size: {len(df)} samples")
            print("\nNew class distribution:")
            print(df['class'].value_counts())
            print("\nSample data:")
            print(df.head(3))
            
            
            
            df['class'] = df['class'].map(self.label_map)
            
            min_label = df['class'].min()
            max_label = df['class'].max()
            print(f"\nLabel range: {min_label} to {max_label}")
            if min_label < 0 or max_label >= self.num_labels:
                raise ValueError(f"Invalid label values detected. Labels must be between 0 and {self.num_labels-1}")
            
            texts = df['text'].values
            labels = df['class'].values
            
            train_X, test_X, train_Y, test_Y = train_test_split(
                texts, labels, 
                train_size=0.8,
                shuffle=True,
                random_state=42,
                stratify=labels  
            )
            
            for split_name, split_labels in [("Training", train_Y), ("Testing", test_Y)]:
                min_label = split_labels.min()
                max_label = split_labels.max()
                print(f"\n{split_name} set label range: {min_label} to {max_label}")
                print(f"{split_name} set class distribution:")
                unique, counts = np.unique(split_labels, return_counts=True)
                for label, count in zip(unique, counts):
                    print(f"Class {self.reverse_label_map[label]}: {count} samples ({count/len(split_labels)*100:.1f}%)")
                if min_label < 0 or max_label >= self.num_labels:
                    raise ValueError(f"Invalid label values in {split_name.lower()} set")
            
            return train_X, test_X, train_Y, test_Y
        except FileNotFoundError:
            print("Error: Could not find the CSV file. Please make sure 'Suicide_Detection.csv' is in the correct directory.")
            return None
        except Exception as e:
            print(f"Error loading the dataset: {str(e)}")
            return None

    def prepare_datasets(self, dataset_path):
        train_X, test_X, train_Y, test_Y = self.load_data(dataset_path)
        if train_X is None:
            exit()
        
        print("Preparing datasets")
        
        # Create datasets with tokenization handled in __getitem__
        train_dataset = BertDataset(
            texts=train_X,
            labels=train_Y,
            tokenizer=self.tokenizer,
            max_length=256
        )
        
        test_dataset = BertDataset(
            texts=test_X,
            labels=test_Y,
            tokenizer=self.tokenizer,
            max_length=256
        )

        print("Creating data loaders")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,  
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32,  
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, num_epochs=5):  # Increased epochs
        if (self.train_loader is None or self.test_loader is None):
            raise ValueError("Datasets are not loaded in. Make sure to call prepare_datasets() first")

        if self.num_labels is None:
            raise ValueError("Number of labels not set. Make sure load_data() was called successfully.")

        train_loader = self.train_loader
        test_loader = self.test_loader

        model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            problem_type="single_label_classification"  
        )
        
        # Use a lower learning rate and add weight decay
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        
        # Add learning rate scheduler
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        model.to(device)

        scaler = torch.amp.GradScaler('cuda')

        print("\nStarting training...")
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            # Create progress bar for training
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
            
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Validate labels before forward pass
                # See comment in load_data
                if labels.min() < 0 or labels.max() >= self.num_labels:
                    print(f"\nInvalid labels detected in batch {batch_idx}:")
                    print(f"Label range: {labels.min()} to {labels.max()}")
                    print(f"Expected range: 0 to {self.num_labels-1}")
                    continue
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                
                total_loss += loss.item()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            # Evaluate and save best model
            accuracy = self.evaluate_model(model, test_loader, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                output_dir = f"{self.model_name}_suicide_classifier"
                model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(f"\nNew best model saved to {output_dir} with accuracy: {accuracy:.2f}%")

    def evaluate_model(self, model, test_loader, device):
        print("\nEvaluating model...")
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        eval_pbar = tqdm(test_loader, desc='Evaluation')
        
        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                current_accuracy = 100 * correct / total
                eval_pbar.set_postfix({'accuracy': f'{current_accuracy:.2f}%'})
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Print confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=[self.reverse_label_map[i] for i in range(self.num_labels)]))
        
        return accuracy

    def load_model(self, model_path=None):

        if model_path is None:
            model_path = f"{self.model_name}_suicide_classifier"
        
        print(f"\nLoading model from {model_path}")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

    def predict_text(self, text):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence

def main():
    bert_loader = BertClassifier(model_name="bert-base-uncased")
    bert_loader.prepare_datasets("Suicide_Detection.csv")
    bert_loader.train_model()
    
if __name__ == "__main__":
    main()
    