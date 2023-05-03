import matplotlib.pyplot as plt

time = []
name = []
linenum = 0;

with open('results.txt', 'r') as f:
    if (linenum > 4):
        for line in f:
            if not line.startswith(' '):
                continue
            cleanline = line.strip()
            arr = cleanline.split("\\s")
            time.append(arr[0])
            name.append(arr[6])
    linenum += 1

plt.pie(time, labels=name, autopct='%1.1f%%')

plt.title('Function Execution Times')

plt.show()
