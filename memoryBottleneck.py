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
    if (staticMem[i] != ''):
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
leg = plt.legend(loc='lower center')
plt.savefig("timestat", dpi='figure', format=None)
plt.clf()


timedynamX = {}
timedynamY = {}
for i in range(len(name)):
    if (dynamicMem[i] != ''):
        if name[i] in timedynamX:
            timedynamX[name[i]].append(float(time[i]))
        else:
            timedynamX[name[i]] = [float(time[i])]
        
        if name[i] in timedynamY:
            timedynamY[name[i]].append(float(dynamicMem[i]))
        else:
            timedynamY[name[i]] = [float(dynamicMem[i])]

default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'm', 'c']))


plt.rc('axes', prop_cycle=default_cycler)

for each in timedynamX:
    plt.plot(timedynamX[each], timedynamY[each], label = each)

plt.title('dynamicMem vs Time')
plt.ylabel('dynamicMem')
plt.xlabel('Time(s)')
leg = plt.legend(loc='lower center')
plt.savefig("timedynam", dpi='figure', format=None)
plt.clf()


timesizeX = {}
timesizeY = {}
for i in range(len(name)):
    if (size[i] != ''):
        if name[i] in timesizeX:
            timesizeX[name[i]].append(float(time[i]))
        else:
            timesizeX[name[i]] = [float(time[i])]
        
        if name[i] in timesizeY:
            timesizeY[name[i]].append(float(size[i]))
        else:
            timesizeY[name[i]] = [float(size[i])]

default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'm', 'c']))


plt.rc('axes', prop_cycle=default_cycler)

for each in timesizeX:
    plt.plot(timesizeX[each], timesizeY[each], label = each)

plt.title('size vs Time')
plt.ylabel('size')
plt.xlabel('Time(s)')
leg = plt.legend(loc='lower center')
plt.savefig("timesize", dpi='figure', format=None)
plt.clf()


timethruX = {}
timethruY = {}
for i in range(len(name)):
    if (throughput[i] != ''):
        if name[i] in timethruX:
            timethruX[name[i]].append(float(time[i]))
        else:
            timethruX[name[i]] = [float(time[i])]
        
        if name[i] in timethruY:
            timethruY[name[i]].append(float(throughput[i]))
        else:
            timethruY[name[i]] = [float(throughput[i])]

default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'm', 'c']))


plt.rc('axes', prop_cycle=default_cycler)

for each in timethruX:
    plt.plot(timethruX[each], timethruY[each], label = each)

plt.title('throughput vs Time')
plt.ylabel('throughput')
plt.xlabel('Time(s)')
leg = plt.legend(loc='lower center')
plt.savefig("timethru", dpi='figure', format=None)
plt.clf()