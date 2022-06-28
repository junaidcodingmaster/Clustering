import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Getting terminal size and making divider for any screen.
sp = "-" * os.get_terminal_size().columns

# Getting data for CSV file.
df = pd.read_csv("gravity.csv")

# Printing divider and some data.
print("\nData Of Gravity and they Stars", sp)
print(df.head())
print(sp)

# Finding K-Mean of data.
x = df.iloc[:, [2, 3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

print(
    "\nGraph -> 'The Elbow Method Graph' or 'The Elbow Point Graph' -> \033[1;32;40m Plotted ! \033[0;0;40m\n"
)

# Plotting 'The Elbow Method Graph' or 'The Elbow Point Graph' .
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker="o", color="red")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

print("\033[0;30;47m Made By Junaid .\033[0m", "\n")
