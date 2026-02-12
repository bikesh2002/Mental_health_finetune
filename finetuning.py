from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import os
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. HARDWARE & PATH SETUP
# ==========================================
# Ensure this filename matches your converted JSONL file
input_file = "mental_health_chat_finetune.jsonl" 
output_dir = "outputs"
max_seq_length = 2048 # Balanced for memory/context
load_in_4bit = True   # Mandatory for consumer GPUs (RTX 3060/4060 etc)

print(">>> [LOG] LOADING MODEL AND TOKENIZER...")

# ==========================================
# 2. MODEL LOADING & LORA OPTIMIZATION
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

# GUARD: Adding Dropout and Weight Decay to prevent "memorization"
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank of 16 for complex therapeutic language
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.05, # Generalization booster
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. DATASET HANDLING & VALIDATION SPLIT
# ==========================================
print(">>> [LOG] PREPARING DATASET...")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")

# SPLIT: 90% Training / 10% Validation to detect overfitting
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (FULL PRODUCTION)
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
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Total Batch Size = 8
        warmup_steps = 5,
        
        # --- FULL RUN SETTINGS ---
        max_steps = 0,               # Disables the 100-step limit
        num_train_epochs = 1,        # Processes the ENTIRE 12k dataset once
        
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        
        # PROTECTION: NOISY GRADIENT CLIPPING
        max_grad_norm = 1.0, 
        
        # AUTOMATED CHECKPOINTING
        save_strategy = "steps",
        save_steps = 100,            # Save progress every 100 steps
        save_total_limit = 2,        # Keep only the last 2 checkpoints to save disk space
        load_best_model_at_end = True, 
        
        # LOGGING & MONITORING
        logging_steps = 1,
        evaluation_strategy = "steps",
        eval_steps = 50,             # Check validation loss every 50 steps
        output_dir = output_dir,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# ==========================================
# 5. START / RESUME TRAINING
# ==========================================
last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        last_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f">>> [LOG] RESUMING FROM PREVIOUS CHECKPOINT: {last_checkpoint}")

print(">>> [LOG] TRAINING COMMENCED. THIS MAY TAKE SEVERAL HOURS...")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. GRAPHING LOSS FUNCTIONS
# ==========================================
print("\n>>> [LOG] GENERATING PERFORMANCE REPORT...")
log_history = trainer.state.log_history
t_steps, t_loss = [e["step"] for e in log_history if "loss" in e], [e["loss"] for e in log_history if "loss" in e]
e_steps, e_loss = [e["step"] for e in log_history if "eval_loss" in e], [e["eval_loss"] for e in log_history if "eval_loss" in e]

plt.figure(figsize=(10, 5))
plt.plot(t_steps, t_loss, label="Training Loss (Learning Patterns)", color="#1f77b4")
plt.plot(e_steps, e_loss, label="Validation Loss (Generalization)", color="#ff7f0e", marker='o')
plt.title("Llama 3 Therapy Model: Learning vs. Generalization")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("final_training_report.png")

# ==========================================
# 7. FINAL EXPORT (SAVING GGUF)
# ==========================================
print("\n>>> [LOG] EXPORTING TO GGUF (4-BIT QUANTIZED)...")
model.save_pretrained_gguf("final_mental_health_model", tokenizer, quantization_method = "q4_k_m")

print("\n>>> [LOG] PIPELINE FINISHED. CHECK 'final_training_report.png' AND THE 'final_mental_health_model' FOLDER.")
