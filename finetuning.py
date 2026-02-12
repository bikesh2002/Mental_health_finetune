import os
import gc
import shutil
import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ==========================================
# 0. SYSTEM & ENVIRONMENT FIXES
# ==========================================
# Prevents disk-limit crashes and silences WandB
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
# 2. MODEL & LORA INITIALIZATION
# ==========================================
print(">>> [LOG] LOADING MODEL AND APPLYING LORA PATCHES...")
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
# 3. DATASET HANDLING & TEMPLATING
# ==========================================
print(">>> [LOG] PREPARING MENTAL HEALTH DATASET...")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")

# 10% validation split helps monitor the model's empathy/emulation balance
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (KAGGLE STABLE)
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
        # 2026 VERSION FIXES
        average_tokens_across_devices = False, 
        eval_strategy = "steps", 
        
        # MEMORY PROTECTION: Batch 1 + Accum 8 fits 15GB VRAM
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True,
        
        warmup_steps = 5,
        num_train_epochs = 1,        
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_grad_norm = 1.0, 
        
        # SAVING: Checkpoints every 100 steps to survive time-outs
        save_strategy = "steps",
        save_steps = 100,            
        save_total_limit = 2,        
        load_best_model_at_end = True, 
        eval_steps = 50,             
        
        report_to = "none", 
        logging_steps = 1,
        output_dir = output_dir,
        optim = "paged_adamw_8bit", # Handles large-dataset memory spikes
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
        print(f">>> [LOG] RESUMING FROM PREVIOUS CHECKPOINT: {last_checkpoint}")

print(">>> [LOG] TRAINING COMMENCED. TARGET: 12,500 ROWS.")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. PERFORMANCE VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 5))
log_history = trainer.state.log_history
t_steps, t_loss = [e["step"] for e in log_history if "loss" in e], [e["loss"] for e in log_history if "loss" in e]
plt.plot(t_steps, t_loss, label="Training Loss")
plt.title("Llama-3 Mental Health Model: Training Progress")
plt.savefig("final_training_report.png")

# ==========================================
# 7. GGUF EXPORT (DISK SPACE WORKAROUND)
# ==========================================
# Export in /tmp to avoid Kaggle's 20GB limit on /working
tmp_path = "/tmp/gguf_final"
final_working_dir = "/kaggle/working/final_model"

print("\n>>> [LOG] EXPORTING TO GGUF VIA SCRATCH SPACE...")
model.save_pretrained_gguf(
    tmp_path, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

# Native Python move (replacement for bash !cp)
if not os.path.exists(final_working_dir):
    os.makedirs(final_working_dir)

for filename in os.listdir(tmp_path):
    if filename.endswith(".gguf") or filename.endswith(".json"):
        shutil.copy(os.path.join(tmp_path, filename), os.path.join(final_working_dir, filename))

print(f"\n>>> [LOG] SUCCESS. MODEL SAVED TO: {final_working_dir}")