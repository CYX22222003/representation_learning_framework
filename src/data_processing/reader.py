import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "./data"
# List all feather files with their sizes
one_hour_files = [
    (f, os.path.getsize(os.path.join(data_dir, f)))
    for f in os.listdir(data_dir)
    if f.endswith("-1h.feather")
]

four_hour_files = [
    (f, os.path.getsize(os.path.join(data_dir, f)))
    for f in os.listdir(data_dir)
    if f.endswith("-4h.feather")
]

one_day_files = [
    (f, os.path.getsize(os.path.join(data_dir, f)))
    for f in os.listdir(data_dir)
    if f.endswith("-1d.feather")
]
# Sort by size descending and take top 50
largest_files_1h = sorted(one_hour_files, key=lambda x: x[1], reverse=True)[:50]
largest_files_4h = sorted(four_hour_files, key=lambda x: x[1], reverse=True)[:50]
largest_files_1d = sorted(one_day_files, key=lambda x: x[1], reverse=True)[:50]

if __name__ == "__main__":
    print(largest_files_1h)
    print(largest_files_4h)
    print(largest_files_1d)
    