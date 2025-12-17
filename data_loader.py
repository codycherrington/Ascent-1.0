import json
import os
from datasets import load_dataset

# Configuration
OUTPUT_FILE = "ascent_data/alpaca_data.json"

def main():
    print("🌍 Loading Alpaca dataset from Hugging Face (tatsu-lab/alpaca)...")
    try:
        # Load the dataset (instruction tuning data)
        dataset = load_dataset("tatsu-lab/alpaca")
    except Exception as e:
        print(f"❌ Failed to load Alpaca: {e}")
        exit(1)

    print(f"✅ Loaded {len(dataset['train'])} examples.")

    formatted_data = []
    print("⚙️  Processing and formatting data...")
    
    for item in dataset['train']:
        # Alpaca has 'instruction', 'input' (optional), and 'output'
        instruction = item['instruction']
        inp = item['input']
        output = item['output']

        # Format matches our model's expectation: 
        # "input": The prompt (Instruction + Context)
        # "output": The response
        
        if inp and len(inp.strip()) > 0:
            full_prompt = f"Input: {instruction}\nContext: {inp}"
        else:
            full_prompt = f"Input: {instruction}"
        
        formatted_data.append({
            "input": full_prompt,
            "output": output
        })

    # Save to JSON
    os.makedirs("ascent_data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"💾 Saved {len(formatted_data)} pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
