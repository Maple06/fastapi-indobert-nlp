import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification

import platform
from ..core.logging import logger

# Defining constants
CWD = os.getcwd()
CATEGORY_LIST = ['Lifestyle', 'Otomotif', 'Music', 'Beauty', 'Fashion', 'Traveling', 'Food', 'Finance', 'Parenting', 'Technology', 'Health', 'Gigs Worker', 'Homedecor', 'Gamers', 'Sport', 'Reviewer', 'Kpop', 'Politik']
ARCHITECTURE = "indobenchmark/indobert-base-p1"

EPOCHS = 10
BATCH_SIZE = 16

defaultEmptyResult = {
                        "username": None,
                        "prediction": None,
                        "category": []
                    }

class NLPIndoBert:
    def __init__(self):
        pass

    def initalTrain(self):
        if os.path.exists(f"{CWD}/ml-models/pytorch_model.bin"):
            logger.info("API Started. Dataset exists.")
            return None
        
        logger.info("API Started. Training dataset...")

        # Set device to use Nvidia GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(f"{CWD}/data/dataset.csv"):
            logger.error("Dataset file does not found. Make sure ./data/dataset.csv exists.")
            return None

        try:
            df = pd.read_csv(f"{CWD}/data/dataset.csv")
            df = df.dropna()
        except:
            logger.error("Dataset csv file have the wrong format. Make sure the csv file only have 2 columns of 'text' and 'label'")
            return None
        
        df['label'] = df['label'].replace(CATEGORY_LIST,range(len(CATEGORY_LIST)))

        # Splitting train dataset into train and validation sets
        train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'], 
                                                                            random_state=2018, 
                                                                            test_size=0.3, 
                                                                            stratify=df['label'])

        val_text, _, val_labels, _ = train_test_split(temp_text, temp_labels, 
                                                                        random_state=2018, 
                                                                        test_size=0.5, 
                                                                        stratify=temp_labels)
        
        # Tokenize and encode sequences in the TRAINING set
        logger.info("Tokenizing and encoding sequences")
        tokenizer = BertTokenizer.from_pretrained(ARCHITECTURE)
        tokens_train = tokenizer.batch_encode_plus(
            train_text.tolist(),
            max_length = 100,
            padding='longest',
            truncation=True
        )

        # Tokenize and encode sequences in the VALIDATION set
        tokens_val = tokenizer.batch_encode_plus(
            val_text.tolist(),
            max_length = 100,
            padding='longest',
            truncation=True
        )

        # Convert lists to tensor
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(train_labels.tolist())

        val_seq = torch.tensor(tokens_val['input_ids'])
        val_mask = torch.tensor(tokens_val['attention_mask'])
        val_y = torch.tensor(val_labels.tolist())

        # Wrap tensors
        train_data = TensorDataset(train_seq, train_mask, train_y)
        val_data = TensorDataset(val_seq, val_mask, val_y)

        # Sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)
        val_sampler = SequentialSampler(val_data)

        # DataLoader for train and validation set
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
        self.val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=BATCH_SIZE)

        # Pass the pre-trained BERT to our define architecture
        self.model = BertForSequenceClassification.from_pretrained(ARCHITECTURE, num_labels=18)
        
        # Push the model to GPU, if exist
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            logger.info("Using device:", torch.cuda.get_device_name(0), flush=True)
        elif self.device == "cpu":
            logger.info("Using device:", platform.processor(), flush=True)

        # Defining optimizer
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-5)

        # Computing class weights
        class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_labels),
                                        y = train_labels                                                 
                                    )
        
        # Converting list of class weights to a tensor
        weights = torch.tensor(class_weights,dtype=torch.float)

        # Push to GPU, if exist
        weights = weights.to(self.device)

        # Set initial loss to infinite
        best_valid_loss = float('inf')

        # Empty lists to store training and validation loss of each epoch
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        for epoch in range(EPOCHS):
            # Train model
            train_loss, train_acc, _ = self.train_epoch(epoch)
            # Evaluate model
            valid_loss, valid_acc, _ = self.evaluate_epoch(epoch)
            
            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.model.save_pretrained(f"{CWD}/ml-models")
            
            # Append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            
            logger.info(f"\nTraining epoch {epoch+1}/{EPOCHS} success. \nTrain acc: {train_acc:.3f} \nTrain loss: {train_loss:.3f} \nValidation acc: {valid_acc:.3f} \nValidation loss: {valid_loss:.3f}")
    
    def clean_sentence(sentence):
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = re.sub('\W', ' ', sentence)
        sentence = re.sub('\s+', ' ', sentence)
        sentence = sentence.strip(' ')
        return sentence

    def train_epoch(self, epoch_num):
        logger.info(f"Training epoch {epoch_num+1}/{EPOCHS}")
        self.model.train()

        total_loss, total_accuracy = 0, 0
        
        # empty list to save model predictions
        total_preds=[]
        
        # iterate over batches
        for step, batch in enumerate(self.train_dataloader):
            # Progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)), flush=True)

            # Push the batch to GPU, if exist
            batch = [r.to(self.device) for r in batch]
        
            sent_id, mask, labels = batch

            # Clear previously calculated gradients 
            self.model.zero_grad()        

            # Get model predictions for the current batch
            preds = self.model(sent_id, mask)

            # Define the loss function
            cross_entropy  = nn.CrossEntropyLoss()

            # Compute the loss between actual and predicted values
            loss = cross_entropy(preds.logits, labels)

            # Add on to the total loss
            total_loss = total_loss + loss.item()

            # Backward pass to calculate the gradients
            loss.backward()

            # Clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters
            self.optimizer.step()

            # Append the model predictions
            total_preds.append(preds)

            # Calculate the accuracy for this batch
            _, pred_labels = torch.max(preds.logits, dim=1)
            accuracy = torch.sum(pred_labels == labels).item() / labels.size(0)
            total_accuracy += accuracy

        # Compute the training accuracy and loss of the epoch
        avg_loss = total_loss / len(self.train_dataloader)
        avg_accuracy = total_accuracy / len(self.train_dataloader)
        
        # Predictions are in the form of (no. of batches, size of batch, no. of classes).
        # Reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate([total_preds], axis=0)

        # Returns the loss, accuracy, and predictions
        return avg_loss, avg_accuracy, total_preds
    
    def evaluate_epoch(self, epoch_num):
        logger.info(f"Evaluating epoch {epoch_num+1}")
        
        # Deactivate dropout layers
        self.model.eval()

        total_loss, total_accuracy = 0, 0
        
        # Empty list to save the model predictions
        total_preds = []

        # Iterate over batches
        for step,batch in enumerate(self.val_dataloader):
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.val_dataloader)), flush=True)

            # Push the batch to GPU, if exist
            batch = [t.to(self.device) for t in batch]

            sent_id, mask, labels = batch

            # Deactivate autograd
            with torch.no_grad():
                # Model predictions
                preds = self.model(sent_id, mask)

                # Define the loss function
                cross_entropy  = nn.CrossEntropyLoss()

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds.logits,labels)

                total_loss = total_loss + loss.item()

                total_preds.append(preds)

            # calculate the accuracy for this batch
            _, pred_labels = torch.max(preds.logits, dim=1)
            accuracy = torch.sum(pred_labels == labels).item() / labels.size(0)
            total_accuracy += accuracy

        # Compute the validation accuracy and loss of the epoch
        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = total_accuracy / len(self.val_dataloader)

        # Reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate([total_preds], axis=0)

        return avg_loss, avg_accuracy, total_preds
    
trainDataset = NLPIndoBert().initalTrain()