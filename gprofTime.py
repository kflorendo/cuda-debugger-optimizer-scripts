import matplotlib.pyplot as plt

time = []
name = []

with open('results.txt', 'r') as f:
    for line in f:
        if not line.startswith(' '):
            continue

        time.append(line[7:14].strip())
        name.append(line[47:].strip())


plt.pie(time, labels=name, autopct='%1.1f%%')

plt.title('Function Execution Times')

plt.show()
