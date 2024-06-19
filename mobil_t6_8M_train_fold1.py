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

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")

else: print("CUDA is not available. Running on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')  # Use 'binary' for binary classification
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define paths to your datasets
data_files = {
    "train": "datasets/hmd_dataset/X_hmd_train_fold1.csv",
    "validation": "datasets/hmd_dataset/X_hmd_val_fold1.csv",
    "test": "datasets/hmd_dataset/X_hmd_test_fold1.csv"
}

# Load your datasets
datasets = load_dataset('csv', data_files=data_files)

# Provide labels
labels_files = {
    "train": "datasets/hmd_dataset/y_mob_train_fold1.csv",
    "validation": "datasets/hmd_dataset/y_mob_val_fold1.csv",
    "test": "datasets/hmd_dataset/y_mob_test_fold1.csv"
}

# Load your labels (assuming binary classification for this example)
labels_datasets = load_dataset('csv', data_files=labels_files)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


# Function to tokenize sequences
def tokenize_function(examples):
    return tokenizer(examples["Protein Sequence"], padding="max_length", truncation=True, max_length=200) #May need to change the max_length (PLM-ARG = 200, originally I did 512)

# Apply tokenization to all splits
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Assuming your datasets and labels are aligned and correctly tokenized
final_datasets = DatasetDict({
    "train": tokenized_datasets["train"].add_column("label", labels_datasets["train"]["Gene Mobility"]),
    "validation": tokenized_datasets["validation"].add_column("label", labels_datasets["validation"]["Gene Mobility"]),
    "test": tokenized_datasets["test"].add_column("label", labels_datasets["test"]["Gene Mobility"])
})

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=2)

# Configure GPU as the device
device = torch.device(f'cuda:{torch.cuda.current_device()}')

# Running on GPU??
model.to(device)  # Move model to the appropriate device
# Print the device where the first parameter tensor of the model is located
print(next(model.parameters()).device)

# Empty the cache for memory
torch.cuda.empty_cache()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mobil_t6_8M_fold1",
    num_train_epochs=10,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    warmup_steps=500,
    save_strategy="epoch",  # Save the model at the end of each epoch to match the evaluation strategy
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Initialize the Trainer
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

metrics_path = os.path.join(output_dir, "mobil_t6_8M_FOLD1_results.txt")
with open(metrics_path, "w") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")
