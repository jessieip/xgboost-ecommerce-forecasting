# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np


data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 92, 78]
}

df = pd.DataFrame(data)

print("--- Environment Test ---")
print(f"Pandas Version: {pd.__version__}")
print(f"Numpy Version: {np.__version__}")
print("\n--- DataFrame Result ---")
print(df)
print("\nMean Score:", df['Score'].mean())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
