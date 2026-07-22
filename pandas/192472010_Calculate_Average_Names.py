import pandas as pd

data = {
    "Name": ["Ram", "Ravi", "Priya"],
    "Marks": [80, 70, 90]
}

df = pd.DataFrame(data)

print("Average Marks:", df["Marks"].mean())