import sys
import glob

import numpy as np

# Define the prefix string to search for
prefix_1 = sys.argv[1]
prefix_2 = sys.argv[2]

print(prefix_1)
print(prefix_2)


def get_folders(prefix):
    folders = glob.glob(prefix + "*")

    return folders


def retrieve_results(folders):
    results_dict = {}
    for folder in folders:
        idx = folder.split("/")[-1]

        try:
            with open(folder + "/evaluate.txt", "r") as f:
                results = f.readlines()[0].strip()
                # print(idx, results)

                idx = idx.split("_")[-1]

                results_dict[idx] = eval(results)
        except:
            print(f"{folder} has no evaluate file.")

    return results_dict


folders_1 = get_folders(prefix_1)
folders_2 = get_folders(prefix_2)

results_1 = retrieve_results(folders_1)
results_2 = retrieve_results(folders_2)

wins_1 = 0

accs_1 = []
accs_2 = []

for n in results_1.keys():

    if n not in results_2:
        continue
    acc_1 = results_1[n]["agg_metrics"]
    acc_2 = results_2[n]["agg_metrics"]

    if acc_1 >= acc_2:
        wins_1 += 1

    accs_1.append(acc_1)
    accs_2.append(acc_2)

print(wins_1)
print(np.mean(accs_1), np.std(accs_1))
print(np.mean(accs_2), np.std(accs_2))
    
