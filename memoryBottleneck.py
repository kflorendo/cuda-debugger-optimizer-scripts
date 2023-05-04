from cycler import cycler
import matplotlib.pyplot as plt

file1 = open(f'output/memoryBottleneckGpuTrace.txt', 'r')
lines = file1.readlines()

time = []
staticMem = []
dynamicMem = []
size = []
throughput = []
name = []
linenum = 0

for line in lines:
    if linenum >= 5:
        cleanline = line.strip()
        lineArr = cleanline.split(',')
        time.append(lineArr[0])
        staticMem.append(lineArr[9])
        dynamicMem.append(lineArr[10])
        size.append(lineArr[11])
        throughput.append(lineArr[12])
        nameArr = lineArr[18:-1]
        name.append(','.join(nameArr))
    linenum += 1


timestatX = {}
timestatY = {}
for i in range(len(name)):
    if name[i] in timestatX:
        timestatX[name[i]].append(time[i])
    else:
        timestatX[name[i]] = [time[i]]
    
    if name[i] in timestatY:
        timestatY[name[i]].append(staticMem[i])
    else:
        timestatY[name[i]] = [staticMem[i]]

default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))


plt.rc('axes', prop_cycle=default_cycler)

for each in timestatX:
    plt.plot(timestatX[each], timestatY[each], label = each)

plt.title('staticMem vs Time')
plt.ylabel('staticMem')
plt.xlabel('Time(s)')
leg = plt.legend(loc='upper right')
plt.savefig("timestat", dpi='figure', format=None)
plt.clf()
