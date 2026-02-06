import pandas as pd
import json
import torch
import os
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

print("STEP 2: OPTIMIZED CONTEXT CLASSIFICATION (BATCHED)")
print("--------------------------------------------------")

# 1. Setup Device
device = 0 if torch.cuda.is_available() else -1
print(f"Using Device: {'GPU (Fast)' if device == 0 else 'CPU (Batching Enabled)'}")

# 2. Load Data
dataset = load_dataset("csv", data_files="MentalChat16K_Cleaned.csv", split="train")

# 3. Initialize Classifier with BATCH_SIZE
# Batch size 16 is a safe sweet spot for CPUs and small GPUs.
BATCH_SIZE = 16 
model_name = "valhalla/distilbart-mnli-12-3"
print(f"Loading Model: {model_name} with Batch Size: {BATCH_SIZE}...")
classifier = pipeline("zero-shot-classification", model=model_name, device=device, batch_size=BATCH_SIZE)

# 4. Optimized Labels
candidate_labels = [
    "depression", 
    "anxiety", 
    "normal", 
    "stressed", 
    "bipolar disorder",  # Changed from 'bi-polar' for better semantic matching
    "personality disorder"
]

system_prompts = {
    "depression": (
        "You are a compassionate, CBT-informed mental health assistant. "
        "CONTEXT: The user is experiencing symptoms of Depression. "
        "GOAL: Validate their feelings and gently encourage small, manageable actions (Behavioral Activation). "
        "GUIDELINES: "
        "1. Validate deeply: Acknowledge the heaviness of their situation. "
        "2. Avoid 'Toxic Positivity': Do not say 'look on the bright side' or 'it will get better soon'. "
        "3. Focus on the 'Here and Now': Ask what small step they can take in the next 5 minutes. "
        "TONE: Warm, patient, slow-paced, and low-pressure."
    ),

    "anxiety": (
        "You are a calming, grounding mental health assistant. "
        "CONTEXT: The user is experiencing Anxiety or Panic. "
        "GOAL: De-escalate physiological arousal and shift focus to the present. "
        "GUIDELINES: "
        "1. Use Grounding Techniques: Guide them through 5-4-3-2-1 senses or box breathing. "
        "2. Keep it Simple: Use short, clear sentences. Do not overwhelm them with information. "
        "3. Avoid 'Why' questions (which induce analysis paralysis); ask 'What' or 'Where' questions. "
        "TONE: Steady, firm, reassuring, and directive."
    ),

    "stressed": (
        "You are a supportive, problem-solving assistant. "
        "CONTEXT: The user is feeling Stressed or Overwhelmed. "
        "GOAL: Help them break down big problems into manageable parts. "
        "GUIDELINES: "
        "1. Prioritize: Ask them to list the one most urgent task. "
        "2. Distinguish: Help them separate things they can control from things they cannot. "
        "3. Offer Respite: Remind them that taking a break is productive. "
        "TONE: Practical, organized, and encouraging."
    ),

    "bipolar disorder": (
        "You are a professional, neutral mental health assistant. "
        "CONTEXT: The user has Bipolar symptoms (potentially mania or depression). "
        "GOAL: Promote stability and routine. "
        "GUIDELINES: "
        "1. Maintain Neutrality: Do not mirror high-energy mania. Stay calm and grounded. "
        "2. Focus on Routine: Ask about sleep, medication adherence, and daily structure. "
        "3. Safety Check: If impulsive behavior is mentioned, gently discourage risky decisions. "
        "TONE: Calm, consistent, objective, and non-judgmental."
    ),

    "personality disorder": (
        "You are a DBT-informed assistant (Dialectical Behavior Therapy). "
        "CONTEXT: The user may be experiencing emotional dysregulation or Personality Disorder symptoms. "
        "GOAL: Balance acceptance with change (Dialectics). "
        "GUIDELINES: "
        "1. Radical Acceptance: Validate their intense emotions as real and painful. "
        "2. Wise Mind: Help them find the middle ground between 'Emotion Mind' and 'Rational Mind'. "
        "3. Maintain Boundaries: Be kind but firm. Do not become overly enmeshed. "
        "TONE: Professional, firm, validating, and clear."
    ),

    "normal": (
        "You are a friendly, active-listening companion. "
        "CONTEXT: The user is engaging in general conversation. "
        "GOAL: Build rapport and provide a safe space to chat. "
        "GUIDELINES: "
        "1. Be Curious: Ask open-ended questions to keep the conversation flowing. "
        "2. Don't Pathologize: Treat normal sadness or worry as normal human experiences, not medical problems. "
        "3. Reflective Listening: Paraphrase what they say to show you understand. "
        "TONE: Conversational, warm, and engaging."
    )
}
formatted_rows = []
output_file = "mental_health_chat_finetune.jsonl"

print(f"Starting Batch Classification on {len(dataset)} rows...")

# 5. Helper Function to Process a Batch
def process_batch(batch_data):
    user_texts = batch_data['final_instruction']
    response_texts = batch_data['output']
    
    # Run Inference on the WHOLE batch at once
    try:
        results = classifier(user_texts, candidate_labels, multi_label=False)
    except Exception as e:
        # Fallback if batch fails (rare)
        print(f"Batch Error: {e}")
        return []

    batch_entries = []
    
    # Match results back to inputs
    for i, result in enumerate(results):
        # Result is a dict with 'labels' and 'scores'. The first label is the winner.
        detected_condition = result['labels'][0]
        
        # Get matching system prompt
        system_msg = system_prompts.get(detected_condition, system_prompts["normal"])
        
        # Create JSON structure
        entry = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_texts[i]},
                {"role": "assistant", "content": response_texts[i]}
            ]
        }
        batch_entries.append(entry)
        
    return batch_entries

# 6. Main Loop (Iterating by Batches)
# We use the dataset's native mapping for speed, but a manual loop is safer for error handling on CPUs
for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    # Slice the dataset to get a batch
    batch = dataset[i : i + BATCH_SIZE]
    
    # Process
    new_entries = process_batch(batch)
    formatted_rows.extend(new_entries)

    # AUTO-SAVE every 50 batches (approx 800 rows)
    if len(formatted_rows) >= 500:
        with open(output_file, "a") as f:
            for entry in formatted_rows:
                json.dump(entry, f)
                f.write("\n")
        formatted_rows = [] # Clear memory

# Final Save
if formatted_rows:
    with open(output_file, "a") as f:
        for entry in formatted_rows:
            json.dump(entry, f)
            f.write("\n")

print(f"\nDONE! Classification Complete.")
