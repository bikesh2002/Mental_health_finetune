from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import os
import matplotlib.pyplot as plt

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
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. DATASET HANDLING & VALIDATION SPLIT
# ==========================================
print(">>> [LOG] PREPARING DATASET...")
# Unified 2026 Unsloth Template call
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
dataset = load_dataset("json", data_files=input_file, split="train")

# 90/10 split for monitoring over-fitting
dataset = dataset.train_test_split(test_size=0.1)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 4. TRAINING ARGUMENTS (2026 STANDARDS)
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
        gradient_accumulation_steps = 4,
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
        
        # Current 2026 argument names
        eval_strategy = "steps", 
        eval_steps = 50,             
        report_to = "none", # Silences W&B
        
        logging_steps = 1,
        output_dir = output_dir,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        seed = 3407,
    ),
)

# ==========================================
# 5. EXECUTION
# ==========================================
last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        last_checkpoint = os.path.join(output_dir, checkpoints[-1])
        print(f">>> [LOG] RESUMING FROM: {last_checkpoint}")

print(">>> [LOG] TRAINING COMMENCED. MONITORING LOSS...")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 6. REPORTING & EXPORT
# ==========================================
plt.figure(figsize=(10, 5))
log_history = trainer.state.log_history
t_steps, t_loss = [e["step"] for e in log_history if "loss" in e], [e["loss"] for e in log_history if "loss" in e]
plt.plot(t_steps, t_loss, label="Training Loss")
plt.savefig("final_training_report.png")

print("\n>>> [LOG] SAVING AS GGUF (Ready for Ollama/Local use)...")
model.save_pretrained_gguf("final_mental_health_model", tokenizer, quantization_method = "q4_k_m")