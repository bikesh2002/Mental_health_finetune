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
# 0. SYSTEM & MEMORY INITIALIZATION
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true" 
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. CONFIGURATION
# ==========================================
input_file = "mental_health_chat_finetune.jsonl" 
output_dir = "/kaggle/working/outputs"
max_seq_length = 2048 
load_in_4bit = True  

# ==========================================
# 2. MODEL & LORA SETUP
# ==========================================
print(">>> [LOG] LOADING MODEL...")
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
# 3. DATASET PREPARATION
# ==========================================
print(">>> [LOG] PROCESSING DATASET...")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (KAGGLE OPTIMIZED)
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
        # 2026 CRITICAL FIXES
        average_tokens_across_devices = False, 
        eval_strategy = "steps", 
        
        # MEMORY MANAGEMENT
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True,
        
        # HYPERPARAMETERS
        warmup_steps = 5,
        num_train_epochs = 1,        
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_grad_norm = 1.0, 
        
        # SAVING & LOGGING
        save_strategy = "steps",
        save_steps = 100,            
        save_total_limit = 2,        
        load_best_model_at_end = True, 
        eval_steps = 50,             
        report_to = "none", 
        logging_steps = 1,
        output_dir = output_dir,
        optim = "paged_adamw_8bit", 
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# ==========================================
# 5. TRAINING EXECUTION
# ==========================================
last_checkpoint = None
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        last_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f">>> [LOG] RESUMING FROM: {last_checkpoint}")

print(">>> [LOG] TRAINING COMMENCED...")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. GGUF EXPORT (DISK SPACE WORKAROUND)
# ==========================================
# Use /tmp for intermediate 16-bit merge files to avoid 20GB limit
tmp_path = "/tmp/gguf_export"

print("\n>>> [LOG] EXPORTING TO GGUF VIA SCRATCH SPACE...")
model.save_pretrained_gguf(
    tmp_path, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

# Move only the final quantized GGUF and config to working directory
!mkdir -p /kaggle/working/final_model
!cp {tmp_path}/*.gguf /kaggle/working/final_model/
!cp {tmp_path}/*.json /kaggle/working/final_model/

print("\n>>> [LOG] SUCCESS. MODEL SAVED TO /kaggle/working/final_model")