import json
import subprocess
import os

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Loop over each outputs_* folder inside iterations/
iterations_root = "iterations"
for folder_name in sorted(os.listdir(iterations_root)):
    if not folder_name.startswith("outputs_"):
        continue

    output_dir = os.path.join(iterations_root, folder_name)
    result_path = os.path.join("results", f"{folder_name}.json")
    result_entries = []

    print(f"\n📂 Evaluating prompts in: {output_dir}")

    # Track per (prompt_type, input_tag)
    type_tag_counters = {}

    for item in problems:
        prompt_type = item["type"]
        input_tag = item.get("prompt_input_type", "text")
        tag_key = (prompt_type, input_tag)

        if tag_key not in type_tag_counters:
            type_tag_counters[tag_key] = 1

        count = type_tag_counters[tag_key]

        for iteration in range(1, 2):  # Only one file per problem
            filename = f"{prompt_type}_{input_tag}_{count:02d}_{iteration:02d}.txt"
            filepath = os.path.join(output_dir, filename)

            if not os.path.exists(filepath):
                print(f"[!] Skipping missing file: {filepath}")
                continue

            print(f"\n--- Evaluating {filename} ---")

            # Call ReasonEval and capture output to memory
            completed = subprocess.run(
                [
                    "python", "evaluate_reasoning.py",
                    "--question", item["content"],
                    "--responses_dir", output_dir,
                    "--model_name_or_path", "GAIR/ReasonEval-7B",
                    "--model_size", "7B",
                    "--prompt_type", prompt_type,
                    "--input_tag", input_tag,
                    "--problem_number", str(count),
                    "--iteration", str(iteration)
                ],
                capture_output=True,
                text=True
            )

            eval_json_path = "eval_results.json"
            if os.path.exists(eval_json_path):
                with open(eval_json_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        result_entry = json.loads(lines[-1])
                        result_entries.append(result_entry)

                # Clear eval_results.json to prevent mixing runs
                with open(eval_json_path, "w") as f:
                    f.write("")

        type_tag_counters[tag_key] += 1

    # Save collected results to results/outputs_X.json
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_entries, f, indent=2)

    print(f"✅ Saved results to {result_path}")
