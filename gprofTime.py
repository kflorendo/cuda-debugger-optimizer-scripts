import matplotlib.pyplot as plt

time = []
name = []
linenum = 0;

for line in open('results.txt', 'r'):
    if (linenum > 5):
        if line.startswith(' '):
            break
        cleanline = line.strip()
        arr = cleanline.split("\\s")
        time.append(arr[0])
        name.append(arr[6])
    linenum += 1

plt.pie(time, labels=name)

plt.title('Function Execution Times')

plt.savefig("gprofTime", dpi='figure', format=None)
