import numpy as np
import os
import pandas as pd
import re
import torch
import wordninja
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def clean_sentence(sent):
    sent = sent.lower()
    sent = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        sent,
        flags=re.MULTILINE,
    )  # Remove URLs
    sent = re.sub(r"\@\w+|\#", "", sent)  # Remove mentions and hashtags
    sent = re.sub(r"\d+", "", sent)  # Remove numbers

    new_words = []
    for word in sent.split(" "):
        word_splits = wordninja.split(word)
        if len(word_splits) > 1:
            new_word_splits = []
            for split_word in word_splits:
                punctuation = ['.', ',', '!', '?']
                if (split_word.isspace() == False and split_word not in punctuation):
                    new_word_splits.append(split_word)
                elif split_word in punctuation and len(new_word_splits) > 0:
                    new_word_splits[-1] += split_word
            word_splits = new_word_splits

        new_words.append(" ".join(word_splits))

    sent = " ".join(new_words)
    return sent


def clean_text(text, num_workers=4):
    with Pool(num_workers) as pool:
        cleaned_text = list(
            tqdm(pool.imap(clean_sentence, text), total=len(text), desc="Cleaning Text")
        )

    return cleaned_text


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
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",  
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0), 
            "attention_mask": encoding["attention_mask"].squeeze(
                0
            ),  
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)


class BertClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.num_labels = None
        self.label_map = None
        self.model = None
        print(f"\nUsing model: {model_name}")

    def collate_fn(self, batch):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"].detach().clone() for item in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"].detach().clone() for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }


    def load_data(self, dataset_path, additional_training_samples=None):
        try:
            df = pd.read_csv(dataset_path, names=["text", "class"]).reset_index(
                drop=True
            )
            df = df.drop(index=0)

            print(f"\nFinal dataset size: {len(df)} samples")
            print("\nFinal class distribution:")
            print(df["class"].value_counts())

            unique_labels = sorted(df["class"].unique())
            self.num_labels = len(unique_labels)
            print(f"\nNumber of unique classes: {self.num_labels}")
            print("Classes:", unique_labels)

            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_map = {
                idx: label for label, idx in self.label_map.items()
            }
            print("\nLabel mapping:", self.label_map)

            class_0_samples = df[df["class"] == self.reverse_label_map[0]]
            class_1_samples = df[df["class"] == self.reverse_label_map[1]]

            total_samples = len(df["class"])

            test_size = int(0.2 * total_samples)
            test_size_0 = int(0.9 * test_size)
            test_size_1 = test_size - test_size_0

            test_df_0 = class_0_samples.sample(n=test_size_0, random_state=42)
            test_df_1 = class_1_samples.sample(n=test_size_1, random_state=42)

            test_df = pd.concat([test_df_0, test_df_1]).sample(frac=1, random_state=42)

            remaining = df.drop(test_df.index)
            remaining_0 = remaining[remaining["class"] == self.reverse_label_map[0]]
            remaining_1 = remaining[remaining["class"] == self.reverse_label_map[1]]

            train_size = min(total_samples - test_size, int(len(remaining_0) / 0.5))
            train_size_0 = int(train_size * 0.5)
            train_size_1 = train_size - train_size_0

            train_df_0 = remaining_0.sample(n=train_size_0, random_state=42)
            train_df_1 = remaining_1.sample(n=train_size_1, random_state=42)
            train_df = pd.concat([train_df_0, train_df_1]).sample(
                frac=1, random_state=42
            )

            test_df["class"] = test_df["class"].map(self.label_map)
            train_df["class"] = train_df["class"].map(self.label_map)

            print(f"\nTrain dataset size: {len(train_df)} samples")
            print("\nTrain dataset class distribution:")
            print(train_df["class"].value_counts())
            print("\nSample data:")
            print(train_df.head(3))

            # Load additional training samples if provided
            if additional_training_samples:
                additional_dfs = []
                for sample_file in additional_training_samples:
                    try:
                        print(f"Loading additional samples from {sample_file}")
                        additional_df = pd.read_csv(
                            sample_file, names=["text", "class"]
                        ).reset_index(drop=True)
                        additional_df = additional_df.drop(index=0)
                        print(additional_df.head(3))

                        temp_unique_labels = sorted(additional_df["class"].unique())
                        temp_label_map = {
                            label: idx for idx, label in enumerate(temp_unique_labels)
                        }
                        additional_df["class"] = additional_df["class"].map(
                            temp_label_map
                        )
                        additional_dfs.append(additional_df)
                        print(f"\nLoaded additional samples from {sample_file}")
                        print(f"Additional samples size: {len(additional_df)}")
                        print("Additional samples class distribution:")
                        print(additional_df["class"].value_counts())
                    except Exception as e:
                        print(
                            f"Warning: Could not load additional samples from {sample_file}: {str(e)}"
                        )

                if additional_dfs:
                    additional_df = pd.concat(additional_dfs, ignore_index=True)
                    train_df = pd.concat([train_df, additional_df], ignore_index=True)
                    print(
                        "\nCombined dataset size after adding additional samples:",
                        len(train_df),
                    )
                    print("Combined dataset class distribution:")
                    print(train_df["class"].value_counts())

            train_X = train_df["text"].values
            train_X = clean_text(train_X)
            train_Y = train_df["class"].values
            test_X = test_df["text"].values
            test_X = clean_text(test_X)
            test_Y = test_df["class"].values

            print(f"\nTest dataset size: {len(test_df)} samples")
            print("\nTest dataset class distribution:")
            print(test_df["class"].value_counts())
            print("\nSample data:")
            print(test_df.head(10))

            return train_X, test_X, train_Y, test_Y
        except FileNotFoundError:
            print(
                "Error: Could not find the CSV file. Please make sure 'Suicide_Detection.csv' is in the correct directory."
            )
            return None
        except Exception as e:
            print(f"Error loading the dataset: {str(e)}")
            return None

    def prepare_datasets(self, dataset_path, additional_training_samples=None):
        train_X, test_X, train_Y, test_Y = self.load_data(
            dataset_path, additional_training_samples
        )
        if train_X is None:
            exit()

        print("Preparing datasets")

        train_dataset = BertDataset(
            texts=train_X, labels=train_Y, tokenizer=self.tokenizer, max_length=128
        )

        test_dataset = BertDataset(
            texts=test_X, labels=test_Y, tokenizer=self.tokenizer, max_length=128
        )

        print("Creating data loaders")

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, num_epochs=5):  
        if self.train_loader is None or self.test_loader is None:
            raise ValueError(
                "Datasets are not loaded in. Make sure to call prepare_datasets() first"
            )

        if self.num_labels is None:
            raise ValueError(
                "Number of labels not set. Make sure load_data() was called successfully."
            )

        train_loader = self.train_loader
        test_loader = self.test_loader

        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
        )

        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        model.to(device)

        scaler = torch.amp.GradScaler("cuda")

        print("\nStarting training...")
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
            )

            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() 

                total_loss += loss.item()

                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            accuracy = self.evaluate_model(model, test_loader, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                output_dir = f"{self.model_name}_suicide_classifier"
                model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(
                    f"\nNew best model saved to {output_dir} with accuracy: {accuracy:.2f}%"
                )

    def evaluate_model(self, model, test_loader, device):
        print("\nEvaluating model...")
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        eval_pbar = tqdm(test_loader, desc="Evaluation")

        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                current_accuracy = 100 * correct / total
                eval_pbar.set_postfix({"accuracy": f"{current_accuracy:.2f}%"})

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(
            classification_report(
                all_labels,
                all_predictions,
                target_names=[
                    self.reverse_label_map[i] for i in range(self.num_labels)
                ],
            )
        )

        return accuracy

    def load_model(self, model_path=None):

        if model_path is None:
            model_path = f"{self.model_name}_suicide_classifier"

        print(f"\nLoading model from {model_path}")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def predict_text(self, text):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()

        text = clean_sentence(text)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = self.tokenizer(
            text, truncation=True, padding=True, max_length=128, return_tensors="pt"
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
    bert_loader.prepare_datasets(
        "Suicide_Detection.csv", ["additional_training_samples.csv"]
    )
    bert_loader.train_model()


if __name__ == "__main__":
    main()
