from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def load_dataset(file_path, tokenizer, block_size=128):
  return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

def train_gpt(file_path):
  model = GPT2LMHeadModel.from_pretrained("gpt2")
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  dataset = load_dataset(file_path, tokenizer)
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  training_args = TrainingArguments(
    output_dir='./models',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=20,
    save_steps=10_000,
    save_total_limit=2
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
  )

  trainer.train()

  model.save_pretrained("./models/gpt-model")
  tokenizer.save_pretrained("./models/gpt-model")