from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import pandas as pd
import numpy as np

# 1. Загрузка и подготовка данных
df = pd.read_csv("../data/balanced_spam_ham.csv")
dataset = Dataset.from_pandas(df[['message', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 2. Загрузка модели и токенизатора
model_name = "super-apple/spam-classifier-ru"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Токенизация
def tokenize_function(examples):
    return tokenizer(examples["message"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["message"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# 4. Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions), "f1": f1_score(labels, predictions)}

# 5. Обучение
training_args = TrainingArguments(
    output_dir="../models",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("../models/my_ruSpam_model")
tokenizer.save_pretrained("../models/my_ruSpam_model")