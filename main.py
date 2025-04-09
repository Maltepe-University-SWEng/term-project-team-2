
# Fine-tune Mistral 7B with LoRA using Unsloth (fikra-cleaned dataset)

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from datasets import load_dataset
import torch


def main():
    # 1. Model ve tokenizer'ı yükle
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/mistral-7b-bnb-4bit",  # 4-bit Mistral
        max_seq_length = 2048,
        dtype = torch.float16,
        load_in_4bit = True
    )

    # 2. LoRA ekle (sadece LoRA katmanlarını eğiteceğiz)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "v_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    # 3. Dataset yükle (fikralar_cleaned.json dosyasını aynı dizine koymalısın)
    dataset = load_dataset("json", data_files="fikralar_cleaned.json")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # 4. Tokenize et
    MAX_LENGTH = 512

    def tokenize(example):
        return tokenizer(
            example["content"],
            truncation = True,
            max_length = MAX_LENGTH,
            padding = "max_length",
        )

    dataset = dataset.map(tokenize, batched=True)

    # 5. Eğitim argümanları
    targs = TrainingArguments(
        output_dir = "fikra-mistral-lora",
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 500,
        learning_rate = 2e-4,
        bf16 = False,
        fp16 = True,
        logging_steps = 10,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        logging_dir = "logs",
        report_to = "none"
    )

    # 6. Eğitimi başlat
    FastLanguageModel.train(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        args = targs
    )

    model.save_pretrained("fikra-mistral-lora")
    print("✅ Eğitim tamamlandı ve model kaydedildi.")



if __name__ == "__main__":
    main()
