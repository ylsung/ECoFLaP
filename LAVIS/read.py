import sys
import glob

# Define the prefix string to search for
prefix = sys.argv[1]

# Use glob to find all folders matching the prefix
folders = glob.glob(prefix + "*")

print(folders)

for folder in folders:
    idx = folder.split("/")[-1]

    try:
        with open(folder + "/evaluate.txt", "r") as f:
            results = f.readlines()[-1].strip()
            print(idx, results)
    except:
        print(f"{folder} has no evaluate file.")

