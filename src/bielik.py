from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Wczytywanie danych z pliku tekstowego
with open('path/to/your_dataset.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Przekształcenie danych w listę słowników i utworzenie obiektu Dataset
dataset = Dataset.from_dict({'text': [line.strip() for line in lines]})

# Podział na zbiory treningowy i walidacyjny
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Tokenizacja danych
tokenizer = AutoTokenizer.from_pretrained("speakleash/Bielik-7B-Instruct-v0.1")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Definiowanie modelu i argumentów treningowych
model = AutoModelForCausalLM.from_pretrained("speakleash/Bielik-7B-Instruct-v0.1")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trenowanie modelu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)
trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()
print(results)

# Zapisanie wytrenowanego modelu
model.save_pretrained("../models")
tokenizer.save_pretrained("../models")
