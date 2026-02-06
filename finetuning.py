from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template_tokenizer_formats_func
import os
import json
import matplotlib.pyplot as plt

# ==========================================
# 1. HARDWARE & PATH SETUP
# ==========================================
input_file = "mental_health_chat_finetune.jsonl" 
output_dir = "outputs"
max_seq_length = 2048 # Balanced for performance and memory
load_in_4bit = True   # Mandatory for consumer GPUs

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

# GUARD: Adding Dropout and Weight Decay to prevent overfitting
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank of 16 is robust for complex therapeutic language
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.05, # Randomly disables 5% of weights to force better generalization
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. DATASET HANDLING & VALIDATION SPLIT
# ==========================================
print(">>> [LOG] PREPARING DATASET...")
tokenizer = get_chat_template_tokenizer_formats_func(tokenizer, format="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")

# SPLIT: 90% Training / 10% Validation to track if model "memorizes" or "learns"
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (THE BRAIN CONTROLS)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"], # Passes the 10% split here
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1, # Kept at 1 for Windows/VS Code stability
    packing = False,
    
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective batch size = 8
        warmup_steps = 5,
        max_steps = 100, # Set to 0 and use 'num_train_epochs = 1' for full 16k dataset run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        
        # PROTECTION: NOISY GRADIENT CLIPPING
        # Prevents outlier data from "shocking" the weights and causing loss spikes
        max_grad_norm = 1.0, 
        
        # AUTOMATED CHECKPOINTING (PAUSE/RESUME)
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = 2,      # Keeps disk usage low
        load_best_model_at_end = True, # REWINDS to the best version if overfitting starts
        
        # LOGGING & MONITORING
        logging_steps = 1,
        evaluation_strategy = "steps",
        eval_steps = 10, # Check validation loss every 10 steps for a detailed graph
        output_dir = output_dir,
        optim = "adamw_8bit", # Optimized for memory
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
        print(f">>> [LOG] RESUMING FROM CHECKPOINT: {last_checkpoint}")

print(">>> [LOG] TRAINING COMMENCED...")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. GRAPHING LOSS FUNCTIONS
# ==========================================
print("\n>>> [LOG] GENERATING LOSS CURVE GRAPH...")
log_history = trainer.state.log_history
t_steps, t_loss = [e["step"] for e in log_history if "loss" in e], [e["loss"] for e in log_history if "loss" in e]
e_steps, e_loss = [e["step"] for e in log_history if "eval_loss" in e], [e["eval_loss"] for e in log_history if "eval_loss" in e]

plt.figure(figsize=(10, 5))
plt.plot(t_steps, t_loss, label="Training Loss (Learning Patterns)", color="#1f77b4")
plt.plot(e_steps, e_loss, label="Validation Loss (Generalization)", color="#ff7f0e", marker='o')
plt.title("Llama 3 Fine-Tuning: Learning vs. Generalization")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("loss_performance_report.png")

# ==========================================
# 7. FINAL EXPORT (SAVING FOR USE)
# ==========================================
print("\n>>> [LOG] EXPORTING MODEL TO GGUF (4-BIT)...")
# This saves the model in a format usable by Ollama, LM Studio, or local apps
model.save_pretrained_gguf("final_mental_health_model", tokenizer, quantization_method = "q4_k_m")

print("\n>>> [LOG] PIPELINE COMPLETE. GRAPH AND GGUF MODEL SAVED.")