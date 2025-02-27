import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,AutoConfig
import datasets


model_name = "."
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Đặt pad token để tránh lỗi khi padding

# Load data
data = datasets.load_dataset("json", data_files="data.jsonl", split="train")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

tokenized_datasets = data.map(tokenize_function, batched=True)
config = AutoConfig.from_pretrained("./config.json")


model = AutoModelForCausalLM.from_config(config)

training_args = TrainingArguments(
    output_dir="output_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10,
    report_to="none"
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)


trainer.train()

model.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")
