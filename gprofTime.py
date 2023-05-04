import matplotlib.pyplot as plt

time = []
name = []
linenum = 0;

for line in open('output/results.txt', 'r'):
    if (linenum == 4):
        index = line.find("name")
    if (linenum > 4):
        if line in ['\n', '\r\n']:
            break
        cleanline = line.strip()
        arr = line.split()
        time.append(arr[0])
        name.append(line[index:].strip())
    linenum += 1

length = len(time)
for i in range(length):
    if (time[length - i - 1] == '0.00'):
        time.pop(length - i - 1)
        name.pop(length - i - 1)

res = [i + '(' + j + ')' for i, j in zip(name, time)]

plt.pie(time, labels=res)

plt.title('Function Execution Times')

plt.savefig("gprofTime", dpi='figure', format=None)
