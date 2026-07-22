import matplotlib.pyplot as plt

subjects = ["Python", "Java", "C", "AI"]
hours = [30, 25, 20, 25]

plt.pie(hours, labels=subjects, autopct="%1.1f%%")
plt.title("Study Hours")
plt.show()