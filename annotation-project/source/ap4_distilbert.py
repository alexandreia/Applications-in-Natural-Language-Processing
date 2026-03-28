import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

# ---------------------------
# Settings
# ---------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "Additional.csv"
DEV_FILE = "Exploration.csv"
TEST_FILE = "Joint.csv"

OUTPUT_DIR = "./distilbert_results"
MAX_LENGTH = 128   # reduced for efficiency
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

FREEZE_BERT = False   # toggle this if you want freezing
DEBUG = False

LABEL_MAP = {
    "APPROPRIATE": 0,
    "LANG": 1,
    "SEX": 1,
}

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------
# Debug preview
# ---------------------------
def inspect_file(path, n_lines=5):
    print(f"\nPreview of {path}:")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            print(repr(line))

# ---------------------------
# Robust CSV loader
# ---------------------------
def load_csv_robust(path):
    candidates = [
        {"sep": ","},
        {"sep": "\t"},
        {"sep": ";"},
    ]

    best_df = None
    best_score = -1

    for cfg in candidates:
        try:
            df = pd.read_csv(path, sep=cfg["sep"], engine="python")

            cols = set(df.columns.astype(str))
            score = sum(c in cols for c in ["Label", "Text"])

            if df.shape[1] >= 2:
                score += 1

            if score > best_score:
                best_df = df
                best_score = score

        except Exception:
            continue

    if best_df is None:
        raise ValueError(f"Could not parse file: {path}")

    return best_df

# ---------------------------
# Data preparation
# ---------------------------
def prepare_dataframe(df, name):
    df = df.copy()
    df = df[["Text", "Label"]]

    df["Text"] = df["Text"].astype(str).str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    df = df[df["Label"].isin(LABEL_MAP.keys())]
    df = df[df["Text"].str.len() > 0]

    df["label"] = df["Label"].map(LABEL_MAP)
    df = df.rename(columns={"Text": "text"})
    df = df[["text", "label"]].reset_index(drop=True)

    print(f"\n{name} label distribution:")
    print(df["label"].value_counts(normalize=True))

    return df

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
    }

# ---------------------------
# Main
# ---------------------------
def main():
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    if DEBUG:
        inspect_file(TRAIN_FILE)
        inspect_file(DEV_FILE)
        inspect_file(TEST_FILE)

    # Load
    train_df = prepare_dataframe(load_csv_robust(TRAIN_FILE), "train")
    dev_df = prepare_dataframe(load_csv_robust(DEV_FILE), "dev")
    test_df = prepare_dataframe(load_csv_robust(TEST_FILE), "test")

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(dev_df),
        "test": Dataset.from_pandas(test_df),
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset_dict.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    if FREEZE_BERT:
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    # Train
    trainer.train()

    # Evaluate
    val_results = trainer.evaluate(tokenized["validation"])
    test_results = trainer.evaluate(tokenized["test"])

    print("\nValidation:", val_results)
    print("Test:", test_results)

    # Random baseline
    y_true = np.array(test_df["label"])
    rng = np.random.default_rng(SEED)
    y_pred = rng.integers(0, 2, size=len(y_true))

    print("\nRandom baseline accuracy:", accuracy_score(y_true, y_pred))

    # ---------------------------
    # Plotting (FIXED)
    # ---------------------------
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    eval_acc_pairs = []

    for entry in log_history:
        if "loss" in entry and "step" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

        if "eval_accuracy" in entry and "step" in entry:
            eval_acc_pairs.append((entry["step"], entry["eval_accuracy"]))

    # Loss plot
    plt.figure()
    plt.plot(train_steps, train_losses, label="Train loss")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

    # Accuracy plot
    if eval_acc_pairs:
        steps, accs = zip(*eval_acc_pairs)
        plt.figure()
        plt.plot(steps, accs, marker="o")
        plt.title("Validation Accuracy")
        plt.show()


if __name__ == "__main__":
    main()
