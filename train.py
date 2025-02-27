from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import datasets
from transformers import AutoConfig
model_name = "."  # Thư mục chứa tokenizer và config.json
tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_file="tokenizer.model")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Dùng EOS token làm padding
    # Tải dữ liệu
data = datasets.load_dataset("json", data_files="data.jsonl", split="train")

# Tokenize dữ liệu
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)

tokenized_datasets = data.map(tokenize_function, batched=True)

# Load mô hình

config = AutoConfig.from_pretrained("config.json")
model = AutoModelForCausalLM.from_config(config)
# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="output_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10
)

# Huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

trainer.train()
