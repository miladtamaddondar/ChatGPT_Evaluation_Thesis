import json
import subprocess
import os

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# Track how many times each (type, input_tag) has occurred
type_tag_counters = {}

for item in problems:
    prompt_type = item["type"]
    input_tag = item.get("prompt_input_type", "text")
    tag_key = (prompt_type, input_tag)

    if tag_key not in type_tag_counters:
        type_tag_counters[tag_key] = 1

    count = type_tag_counters[tag_key]

    for iteration in range(1, 2):  # Evaluate 1 repetition instead of the original 10
        filename = f"{prompt_type}_{input_tag}_{count:02d}_{iteration:02d}.txt"
        filepath = os.path.join("outputs", filename)

        if not os.path.exists(filepath):
            print(f"[!] Skipping missing file: {filepath}")
            continue

        print(f"\n--- Evaluating {filename} ---")
        subprocess.run([
            "python", "evaluate_reasoning.py",
            "--question", item["content"],
            "--responses_dir", "outputs",
            "--model_name_or_path", "GAIR/ReasonEval-7B",
            "--model_size", "7B",
            "--prompt_type", prompt_type,
            "--input_tag", input_tag,
            "--problem_number", str(count),
            "--iteration", str(iteration)
        ])

    type_tag_counters[tag_key] += 1
