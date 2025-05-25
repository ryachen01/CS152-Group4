import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class BertLoader():
    def __init__(self, model_name, dataset_path):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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

    def load_data(self):
        try:
            df = pd.read_csv(self.dataset_path, names=['text', 'class']).reset_index(drop=True)
            df = df.sample(150000)
            print(f"Dataset size: {len(df)} samples")
            print("\nSample data:")
            print(df.head(3))
            
            unique_labels = sorted(df['class'].unique())
            self.num_labels = len(unique_labels)
            print(f"\nNumber of unique classes: {self.num_labels}")
            print("Classes:", unique_labels)
            
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
            print("\nLabel mapping:", self.label_map)
            
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
                random_state=42
            )
            
            for split_name, split_labels in [("Training", train_Y), ("Testing", test_Y)]:
                min_label = split_labels.min()
                max_label = split_labels.max()
                print(f"\n{split_name} set label range: {min_label} to {max_label}")
                if min_label < 0 or max_label >= self.num_labels:
                    raise ValueError(f"Invalid label values in {split_name.lower()} set")
            
            return train_X, test_X, train_Y, test_Y
        except FileNotFoundError:
            print("Error: Could not find the CSV file. Please make sure 'Suicide_Detection.csv' is in the correct directory.")
            return None
        except Exception as e:
            print(f"Error loading the dataset: {str(e)}")
            return None

    def prepare_datasets(self):
        train_X, test_X, train_Y, test_Y = self.load_data()
        if train_X is None:
            exit()
        
        print("Tokenizing texts")

        # Process texts in chunks to reduce memory usage
        # Note that this code was written using a LLM to assist in tokenizing 
        # the data using chunks in order to reduce the memory load
        chunk_size = 1000
        train_encodings = []
        test_encodings = []

        for i in range(0, len(train_X), chunk_size):
            print(i)
            chunk_texts = list(train_X[i:i + chunk_size])
            chunk_encodings = self.tokenizer(
                chunk_texts,
                truncation=True,
                padding=False,  
                max_length=256  
            )
            train_encodings.append(chunk_encodings)

        for i in range(0, len(test_X), chunk_size):
            print(i)
            chunk_texts = list(test_X[i:i + chunk_size])
            chunk_encodings = self.tokenizer(
                chunk_texts,
                truncation=True,
                padding=False,  
                max_length=256  
            )
            test_encodings.append(chunk_encodings)

        print("Combining encodings")

        train_encodings = {
            'input_ids': [item for chunk in train_encodings for item in chunk['input_ids']],
            'attention_mask': [item for chunk in train_encodings for item in chunk['attention_mask']]
        }
        test_encodings = {
            'input_ids': [item for chunk in test_encodings for item in chunk['input_ids']],
            'attention_mask': [item for chunk in test_encodings for item in chunk['attention_mask']]
        }

        train_dataset = BertDataset(train_encodings, train_Y)
        test_dataset = BertDataset(test_encodings, test_Y)

        print("Finalizing preparation")

        # Use dynamic padding in the DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32,  
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32,  
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, num_epochs=3): 

        if (self.train_loader is None or self.test_loader is None):
            self.prepare_datasets()

        if self.num_labels is None:
            raise ValueError("Number of labels not set. Make sure load_data() was called successfully.")

        train_loader = self.train_loader
        test_loader = self.test_loader

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            problem_type="single_label_classification"  
        )
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        model.to(device)

        scaler = ttorch.amp.GradScaler('cuda')

        print("\nStarting training...")
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
                
                total_loss += loss.item()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            self.evaluate_model(model, test_loader, device)

    def evaluate_model(self, model, test_loader, device):
        print("\nEvaluating model...")
        model.eval()
        correct = 0
        total = 0
        
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
                
                current_accuracy = 100 * correct / total
                eval_pbar.set_postfix({'accuracy': f'{current_accuracy:.2f}%'})
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

        output_dir = f"{self.model_name}_suicide_classifier"
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to {output_dir}")

    def load_model(self, model_path=None):

        if model_path is None:
            model_path = f"{self.model_name}_suicide_classifier"
        
        print(f"\nLoading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        

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
            
            # Get the predicted class and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class, confidence

if __name__ == "__main__":
    bert_loader = BertLoader(model_name="distilbert-base-uncased", dataset_path="Suicide_Detection.csv")
    # bert_loader.prepare_datasets()
    # bert_loader.train_model()
    
    bert_loader.load_model()
    
    for _ in range(5):
        test_text = "I am feeling very happy today!"
        predicted_class, confidence = bert_loader.predict_text(test_text)
        print(f"\nPrediction for text: '{test_text}'")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    