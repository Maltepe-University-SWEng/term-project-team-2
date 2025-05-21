import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# 1. Hugging Face Token
HF_TOKEN = "hf_your_hugging_face_token"
login(token=HF_TOKEN)

# 2. Model ve Tokenizer
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. JSON Veri Yükle
with open("FikralarFinal.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts = ["<|fıkra|>\n" + entry["metin"] for entry in raw_data if entry.get("metin")]
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
val_data = dataset["test"]

# 4. Tokenize Et
def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
val_data = val_data.map(tokenize, batched=True, remove_columns=["text"])

# 5. LoRA Ayarı
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.train()

# 6. Eğitim Parametreleri
training_args = TrainingArguments(
    output_dir="fikra-turkish-gpt2",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="logs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

print("🤖 Eğitim başlıyor...")
trainer.train()
trainer.save_model("fikra-turkish-gpt2")
print("✅ Eğitim tamamlandı.")

# 7. Test / Örnek Üretim
prompt = "<|fıkra|>\nLütfen özgün ve komik bir fıkra yaz.\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"🤖 Prompt: {prompt}")
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.2,
    )

print("🎉 Üretilen Fıkra:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
