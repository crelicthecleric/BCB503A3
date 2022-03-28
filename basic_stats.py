import pandas as pd

df = pd.read_csv("/home/ejraven/bcb503-a3/1.0combined.csv")

#List all columns in the dataset
print("All columns in dataset")
for col in df.columns:
    print(col)

#List all activities recorded
print("\nAll activities recorded")
activities_list = df["activity"].unique()
for activity in activities_list:
    print(activity)

#Most frequent activity recorded
print("\nAll activities recorded")
df["activity"].mode()

#Average complexity (complexity or measure or entropy in sensor counts)
print("\nAverage complexity (complexity or measure or entropy in sensor counts)")
print(df["complexity"].mean())

#Average number of seconds since the bedroom sensor was last seen
print("\nAverage number of seconds since the bedroom sensor was last seen")
print(df["sensorElTime-Bedroom"].mean())