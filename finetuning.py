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
# 0. PRE-FLIGHT MEMORY CLEARING
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. HARDWARE & PATH SETUP
# ==========================================
input_file = "mental_health_chat_finetune.jsonl" 
output_dir = "outputs"
max_seq_length = 2048 
load_in_4bit = True  

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

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.05, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Crucial for OOM
    random_state = 3407,
)

# ==========================================
# 3. DATASET HANDLING
# ==========================================
print(">>> [LOG] PREPARING DATASET...")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (MEMORY OPTIMIZED)
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
        # MEMORY FIX: Batch size 1 + Accumulation 8 = Effective Batch Size 8
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True, # Saves ~2GB of VRAM
        
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
        
        eval_strategy = "steps", 
        eval_steps = 50,             
        report_to = "none", 
        
        logging_steps = 1,
        output_dir = output_dir,
        optim = "paged_adamw_8bit", # 'paged' handles memory spikes better
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# ==========================================
# 5. EXECUTION
# ==========================================
print(">>> [LOG] TRAINING COMMENCED. THIS VERSION IS TUNED TO AVOID OOM.")
trainer_stats = trainer.train()

# ==========================================
# 6. REPORTING & EXPORT
# ==========================================
plt.figure(figsize=(10, 5))
log_history = trainer.state.log_history
t_steps, t_loss = [e["step"] for e in log_history if "loss" in e], [e["loss"] for e in log_history if "loss" in e]
plt.plot(t_steps, t_loss, label="Training Loss")
plt.savefig("final_training_report.png")

print("\n>>> [LOG] SAVING AS GGUF...")
model.save_pretrained_gguf("final_mental_health_model", tokenizer, quantization_method = "q4_k_m")