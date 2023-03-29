
import json
import sys
import numpy as np

tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb", 
    "squad_v2", "winogrande_debiased", 
    "superglue-record", "superglue-multirc", "superglue-boolq", "superglue-cb", "superglue-multirc", "superglue-wic", "superglue-copa"]

file_prefix = sys.argv[1]

print(file_prefix)

seeds = [0]

print_results = []
for task in tasks:
    metrics = []
    for seed in seeds:
        file_name = f"{file_prefix}{task}@{seed}.json"

        try:
            data = json.load(open(file_name, "r"))

            # print(data.keys())
            metric = data["test_combined_score"] * 100 if task != "squad_v2" else data["test_combined_score"]
        except:
            metric  = 0

        metrics.append(metric)

    print_results.append(float(f"{np.mean(metrics):.2f}"))

    print(f"[{task}] avg: {np.mean(metrics):.2f} std: {np.std(metrics):.2f}")

print(print_results)
avg_metrics = []

for seed in seeds:
    metrics = []
    for task in tasks:
        file_name = f"{file_prefix}{task}@{seed}.json"

        try:
            data = json.load(open(file_name, "r"))

            # print(data.keys())
            metric = data["test_combined_score"] * 100 if task != "squad_v2" else data["test_combined_score"]
        except:
            metric = 0

        metrics.append(metric)

    avg_metrics.append(np.mean(metrics))

print(avg_metrics)

print(f"mean: {np.mean(avg_metrics):.2f} std: {np.std(avg_metrics):.2f}")
