import sys
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

if len(sys.argv) <= 1:
    print("Not enough arguments passed in.")
    exit()

trace = sys.argv[1]

traceStr = ""
if trace == "gpu":
    traceStr = "Gpu"
elif trace == "api":
    traceStr = "Api"
else:
    print(trace)
    print("Unsupported trace type (either 'gpu' or 'api').")
    exit()

file1 = open(f'output/timeBottleneck{traceStr}Trace.txt', 'r')
lines = file1.readlines()

num = 0
catSet = set()
data = []

for line in lines:
    if num >= 5:
        lineArr = line.split(',')
        start = float(lineArr[0])
        duration = float(lineArr[1])
        name = ''
        if trace == "gpu":
            nameArr = lineArr[18:-1]
            name = ','.join(nameArr)
        else:
            name = lineArr[2]
        data.append((start, duration, name))
        catSet.add(name)
    num += 1

catnum = 1
cats = {}
colormapping = {}
yticks = []
yticklabels = []
for cat in catSet:
    cats[cat] = catnum
    colormapping[cat] = f'C{catnum}'
    yticks.append(catnum)
    yticklabels.append(cat)
    catnum += 1

verts = []
colors = []
for d in data:
    # print((d[0], d[1], d[2]))
    v =  [(d[0], cats[d[2]]-.4),
          (d[0], cats[d[2]]+.4),
          (d[0] + d[1], cats[d[2]]+.4),
          (d[0] + d[1], cats[d[2]]-.4),
          (d[0], cats[d[2]]-.4)]
    verts.append(v)
    colors.append(colormapping[d[2]])

bars = PolyCollection(verts, facecolors=colors)

plt.tight_layout()
fig, ax = plt.subplots()
ax.add_collection(bars)
ax.autoscale()
ax.set_xlabel("Time (microseconds)")

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
fig.savefig(f'output/timeBottleneck{traceStr}Timeline.png',dpi=300, bbox_inches = "tight")
