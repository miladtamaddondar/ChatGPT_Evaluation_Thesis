import os
import json
import argparse
import ast
import math
import io
import contextlib
from ReasonEval.codes.examples import get_results

def parse_steps_from_file(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return content

def evaluate_with_reasoneval(question, steps, model_args):
    """Run ReasonEval and capture printed output for parsing."""
    results = {
        "solution_level_validity_scores": float('nan'),
        "solution_level_redundancy_scores": float('nan'),
        "step_level_validity_scores": [],
        "step_level_redundancy_scores": []
    }

    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            get_results(model_args, question, steps)

        output = buffer.getvalue()
        output_lines = output.splitlines()

        for line in output_lines:
            if 'step_level_validity_scores:' in line:
                try:
                    val_str = line.split(':', 1)[-1].strip()
                    results["step_level_validity_scores"] = ast.literal_eval(val_str)
                except Exception as e:
                    print(f"Error parsing step validity: {e}")
            elif 'step_level_redundancy_scores:' in line:
                try:
                    red_str = line.split(':', 1)[-1].strip()
                    results["step_level_redundancy_scores"] = ast.literal_eval(red_str)
                except Exception as e:
                    print(f"Error parsing step redundancy: {e}")
            elif 'solution_level_validity_scores:' in line:
                try:
                    val = float(line.split(':', 1)[-1].strip())
                    if not math.isnan(val):
                        results["solution_level_validity_scores"] = val
                except:
                    pass
            elif 'solution_level_redundancy_scores:' in line:
                try:
                    val = float(line.split(':', 1)[-1].strip())
                    if not math.isnan(val):
                        results["solution_level_redundancy_scores"] = val
                except:
                    pass

    except Exception as e:
        print(f"[ReasonEval wrapper error] {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--responses_dir", type=str, default="outputs")
    parser.add_argument("--model_name_or_path", type=str, default="GAIR/ReasonEval-7B")
    parser.add_argument("--model_size", choices=["7B", "34B"], default="7B")
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--input_tag", choices=["text", "image"], required=True)
    parser.add_argument("--problem_number", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    txt_file = f"{args.prompt_type}_{args.input_tag}_{args.problem_number:02d}_{args.iteration:02d}.txt"
    path = os.path.join(args.responses_dir, txt_file)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        exit(1)

    content = parse_steps_from_file(path)
    steps = [line.strip() for line in content.split('\n') if line.strip()]

    print(f"\n=== Evaluating {txt_file} ===")

    # Run ReasonEval
    re_results = evaluate_with_reasoneval(
        args.question,
        steps,
        argparse.Namespace(
            model_name_or_path=args.model_name_or_path,
            model_size=args.model_size
        )
    ) or {}

    # Save full results
    result_entry = {
        "type": args.prompt_type,
        "prompt_input_type": args.input_tag,
        "problem_number": args.problem_number,
        "iteration": args.iteration,
        "solution_level_validity_score": (
            None if math.isnan(re_results.get("solution_level_validity_scores", float("nan")))
            else re_results["solution_level_validity_scores"]
        ),
        "solution_level_redundancy_score": (
            None if math.isnan(re_results.get("solution_level_redundancy_scores", float("nan")))
            else re_results["solution_level_redundancy_scores"]
        ),
        "step_level_validity_scores": re_results.get("step_level_validity_scores", []),
        "step_level_redundancy_scores": re_results.get("step_level_redundancy_scores", [])
    }

    with open("eval_results.json", "a", encoding="utf-8") as f:
        json.dump(result_entry, f)
        f.write("\n")
