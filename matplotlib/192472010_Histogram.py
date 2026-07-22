import matplotlib.pyplot as plt

data = [12, 15, 18, 20, 20, 25, 30, 35, 40]

plt.hist(data, bins=5)
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()