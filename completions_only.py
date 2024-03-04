from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel


max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# ===== Load base model =====
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # 使用 Flash Attention-2
)

# ===== Add LoRA adapters =====
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# ===== Data Preparation =====
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
    map_eos_token=True,  # Maps <|im_end|> to </s> instead
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


dataset = load_dataset("json", data_files="openhermes2_5.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True,)

# ===== Define Response Template =====
response_template = "\n<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# ===== Train the model =====
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    neftune_noise_alpha=5,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

# ===== Saving finetuned model =====
model.save_pretrained("lora_model_completions")

model.save_pretrained_merged("model_16bit_completions", tokenizer, save_method="merged_16bit",)

if False:
    model.save_pretrained_merged("model_lora_completions", tokenizer, save_method="lora",)
