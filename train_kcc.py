import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("⏳ Step 1: Loading and cleaning the KCC dataset...")
df = pd.read_csv("KCC_Call_Dataset.csv")
df = df.dropna(subset=["questions", "answers"]) # Prevent empty row crashes

# Convert to Hugging Face format and split off a 1% evaluation set
hf_dataset = Dataset.from_pandas(df)
split_data = hf_dataset.train_test_split(test_size=0.01, seed=42)

# Pick a highly capable base architecture (Qwen 1.5B)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("⏳ Step 2: Fetching base tokenizer and model weights...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Step 3: Map the CSV structure into a conversational chat format
def apply_template(batch):
    texts = []
    for q, a in zip(batch["questions"], batch["answers"]):
        conversation = [
            {"role": "system", "content": "You are a Kisan Call Center agricultural expert agent."},
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": str(a)}
        ]
        texts.append(tokenizer.apply_chat_template(conversation, tokenize=False))
    return {"text": texts}

train_set = split_data["train"].map(apply_template, batched=True)
eval_set = split_data["test"].map(apply_template, batched=True)

# Step 4: Configure LoRA Adapters
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Step 5: Configure training args (pure arguments only)
training_args = SFTConfig(
    output_dir="./kcc_agent_checkpoints",
    dataset_text_field="text",       
    max_length=128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,                       # Fast processing on data-center GPUs
    gradient_checkpointing=True,
    use_cache=False,
    logging_steps=10,
    max_steps=300,                   # Fine-tunes general style and content rapidly
    save_strategy="steps",
    save_steps=150,
    report_to="none"
)

print(" Step 6: Initializing the training harness...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=eval_set,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args             
)

print(" Launching fine-tuning run on the node...")
trainer.train()

print(" Saving the final trained agricultural agent weights...")
trainer.model.save_pretrained("./my_kcc_agent")
tokenizer.save_pretrained("./my_kcc_agent")
print(" Setup and training process complete!")