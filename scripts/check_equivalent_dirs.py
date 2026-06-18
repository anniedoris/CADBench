import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir1", type=str)
parser.add_argument("dir2", type=str)
args = parser.parse_args()

files1 = set(os.listdir(args.dir1))
files2 = set(os.listdir(args.dir2))

print(f"Directory 1: {len(files1)} files")
print(f"Directory 2: {len(files2)} files")

if files1 == files2:
    print("Directories contain identical files.")
else:
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    if only_in_1:
        print(f"Only in {args.dir1}: {sorted(only_in_1)}")
    if only_in_2:
        print(f"Only in {args.dir2}: {sorted(only_in_2)}")
