import sys
import glob

# Define the prefix string to search for
prefix = sys.argv[1]

# Use glob to find all files matching the prefix
files = glob.glob(prefix + "*")

print(files)

for file in files:
    idx = file.split("/")[-1]

    try:
        with open(file, "r") as f:
            lines = f.readlines()

            results = ".".join([lines[0].strip(), lines[-1].strip()])
            print(idx, results)
    except:
        print(f"{file} is not valid.")

