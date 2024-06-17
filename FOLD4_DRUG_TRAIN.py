import os
import sys
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModel
from datasets import DatasetDict, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import expit as sigmoid

# Set the environment variables to (hopefully) use more GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

###

'''
def monitor_memory_usage(device=None):
  """
  Monitors memory usage on the specified device (default: GPU if available).

  Args:
      device: The device to monitor memory usage on.

  Returns:
      A dictionary containing the following keys:
          allocated: Current allocated memory in bytes.
          max_allocated: Maximum memory ever allocated in bytes.
          reserved: Total amount of reserved memory in bytes.
          max_reserved: Maximum reserved memory ever allocated in bytes.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  if device == "cuda":
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
  else:
    allocated = torch.get_allocated_memory()
    max_allocated = torch.max_allocated_memory()
    reserved = 0  # Not available on CPU
    max_reserved = 0  # Not available on CPU

  return {
      "allocated": allocated,
      "max_allocated": max_allocated,
      "reserved": reserved,
      "max_reserved": max_reserved,
  }


###
'''

os.chdir('/home/tac0225/Documents/plm')
print(os.getcwd())

# GPU
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")


# Define paths to your datasets
sequence_files = {
    "train": "5fold/X_train_fold4.csv",
    "validation": "5fold/X_val_fold4.csv",
    "test": "5fold/X_test_fold4.csv"
}

labels_files = {
    "train": "5fold/y_train_drug_fold4.csv",
    "validation": "5fold/y_val_drug_fold4.csv",
    "test": "5fold/y_test_drug_fold4.csv"
}

# Load the tokenized sequences
def load_and_tokenize_sequences(sequence_files):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Function to tokenize sequences
    def tokenize_function(examples):
        return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=200) # Originally 256

    tokenized_datasets = {split: load_dataset('csv', data_files={split: file})[split].map(tokenize_function, batched=True)
                          for split, file in sequence_files.items()}
    return tokenized_datasets

# Load the labels
def load_labels(labels_files):
    labels_datasets = {split: pd.read_csv(path).values for split, path in labels_files.items()}
    return labels_datasets

# When preparing your datasets, ensure that a 'labels' tensor is correctly included in your dataset
def prepare_datasets(tokenized_datasets, labels_datasets):
    final_datasets = {}
    for split in tokenized_datasets.keys():
        # Convert the labels into a PyTorch tensor and ensure it matches the sequence dataset length
        labels = torch.tensor(labels_datasets[split]).float()  # Ensure dtype is float for BCELoss
        # Create a new dataset with both inputs and labels
        final_datasets[split] = tokenized_datasets[split].add_column("labels", labels.tolist())
    return DatasetDict(final_datasets)

def compute_metrics(p):
    probs = sigmoid(p.predictions)
    preds = np.where(probs > 0.5, 1, 0)

    # Add zero_division=1 to handle cases where precision or recall could be undefined
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='samples', zero_division=1)
    
    accuracy = accuracy_score(p.label_ids, preds)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Load the model for multi-class classification
model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=19)

# Trainer initialization
training_args = TrainingArguments(
    output_dir="./results_drug_fold4",
    num_train_epochs=20,
    per_device_train_batch_size=10,  # Adjust the batch size according to your GPU capabilities
    per_device_eval_batch_size=10,
    warmup_steps=500,
    weight_decay=0.10,
    evaluation_strategy="epoch",  # Perform evaluation at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch to match the evaluation strategy
    logging_dir="./logs_drug",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True
)

# Load and prepare data
tokenized_datasets = load_and_tokenize_sequences(sequence_files)
labels_datasets = load_labels(labels_files)
final_datasets = prepare_datasets(tokenized_datasets, labels_datasets)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

#Evaluate the model
results = trainer.evaluate(final_datasets["test"])
print(results)

output_dir = './'

metrics_path = os.path.join(output_dir, "results_drug_fold4.txt")
with open(metrics_path, "w") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")
