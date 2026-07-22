import pandas as pd

data = {
    "Name": ["Ram", "Ravi", "Priya"],
    "Marks": [80, 65, 90]
}

df = pd.DataFrame(data)

print(df[df["Marks"] > 70])