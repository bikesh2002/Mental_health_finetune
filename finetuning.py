from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import os
import gc
import matplotlib.pyplot as plt

# ==========================================
# 0. MEMORY & ENVIRONMENT FIXES
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true" # Silences the WandB prompt
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. HARDWARE & PATH SETUP
# ==========================================
input_file = "mental_health_chat_finetune.jsonl" 
output_dir = "outputs"
max_seq_length = 2048 
load_in_4bit = True  

# ==========================================
# 2. MODEL LOADING & LORA OPTIMIZATION
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.05, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# ==========================================
# 3. DATASET HANDLING
# ==========================================
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (UPDATED FOR 2026 FIX)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"], 
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1, 
    packing = False,
    
    args = TrainingArguments(
        # CRITICAL 2026 BUG FIXES:
        average_tokens_across_devices = False, 
        eval_strategy = "steps", # Modern keyword
        
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True,
        
        warmup_steps = 5,
        max_steps = 0,               
        num_train_epochs = 1,        
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_grad_norm = 1.0, 
        save_strategy = "steps",
        save_steps = 100,            
        save_total_limit = 2,        
        load_best_model_at_end = True, 
        eval_steps = 50,             
        report_to = "none", 
        logging_steps = 1,
        output_dir = output_dir,
        optim = "paged_adamw_8bit", # Better memory safety
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# ==========================================
# 5. EXECUTION
# ==========================================
print(">>> [LOG] TRAINING COMMENCED. 2026 COMPATIBILITY PATCHES APPLIED.")
trainer_stats = trainer.train()

# ==========================================
# 6. EXPORT
# ==========================================
model.save_pretrained_gguf("final_mental_health_model", tokenizer, quantization_method = "q4_k_m")