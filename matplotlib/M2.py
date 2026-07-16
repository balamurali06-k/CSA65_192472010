import matplotlib.pyplot as plt

students = ["A", "B", "C", "D"]
marks = [80, 70, 90, 85]

plt.bar(students, marks)
plt.title("Bar Chart")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()