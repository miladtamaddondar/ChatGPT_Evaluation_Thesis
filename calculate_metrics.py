import json
from collections import defaultdict

def load_results():
    results = []
    try:
        with open("eval_results.json", "r") as f:
            for line in f:
                results.append(json.loads(line))
    except FileNotFoundError:
        pass
    return results

def print_metrics(results):
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for r in results:
        metrics[r["type"]]["total"] += 1
        metrics[r["type"]]["correct"] += int(r["correct"])
    
    print("\n=== Accuracy Metrics ===")
    for ptype, vals in metrics.items():
        acc = vals["correct"] / vals["total"]
        print(f"{ptype}: {vals['correct']}/{vals['total']} = {acc:.1%}")
    
    if metrics:
        total_correct = sum(v["correct"] for v in metrics.values())
        total = sum(v["total"] for v in metrics.values())
        print(f"\nOVERALL: {total_correct}/{total} = {total_correct/total:.1%}")

if __name__ == "__main__":
    results = load_results()
    print_metrics(results)