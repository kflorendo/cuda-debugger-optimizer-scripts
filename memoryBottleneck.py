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
    if (time[i] != ""):
        if name[i] in timestatX:
            timestatX[name[i]].append(float(time[i]))
        else:
            timestatX[name[i]] = [float(time[i])]
        
        if name[i] in timestatY:
            timestatY[name[i]].append(float(staticMem[i]))
        else:
            timestatY[name[i]] = [float(staticMem[i])]

default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'm', 'c']))


plt.rc('axes', prop_cycle=default_cycler)

for each in timestatX:
    plt.plot(timestatX[each], timestatY[each], label = each)

plt.title('staticMem vs Time')
plt.ylabel('staticMem')
plt.xlabel('Time(s)')
leg = plt.legend(loc='bottom center')
plt.savefig("timestat", dpi='figure', format=None)
plt.clf()
