import sys
import os

folder = sys.argv[1]

tasks = [
    "caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", 
    "imagenet", "oxford_flowers", "oxford_pets", "stanford_cars", 
    "sun397", "ucf101",
]

accuracies = []
for task in tasks:
    output_file = os.path.join(folder, task, "log.txt")
    
    try:
        with open(output_file, "r") as f:
            output = f.readlines()
        accuracy = output[-3].split(" ")[-1].strip()
    except:
        accuracy = -1
        
    accuracies.append(accuracy)
    
max_width = max(len(str(item)) for item in tasks + accuracies)

for task, acc in zip(tasks, accuracies):
    max_width = max(len(str(task)), len(str(acc)))
    print(f'{str(task):<{max_width}}', end=' ')
print()
for task, acc in zip(tasks, accuracies):
    max_width = max(len(str(task)), len(str(acc)))
    print(f'{str(acc):<{max_width}}', end=' ')