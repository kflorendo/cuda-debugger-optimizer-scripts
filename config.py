import sys
import matplotlib.pyplot as plt

gridX = {}
gridY = {}
gridZ = {}
blockX = {}
blockY = {}
blockZ = {}
linenum = 0;
fileName = sys.argv[1]

for line in open(fileName, 'r'):
    cleanline = line.strip()
    strarr = cleanline.split()
    arr = [float(v) for v in strarr]

    if arr[0] in gridX:
        gridX[arr[0]].append(arr[6])
    else:
        gridX[arr[0]] = [arr[6]]

    if arr[1] in gridY:
        gridY[arr[1]].append(arr[6])
    else:
        gridY[arr[1]] = [arr[6]]

    if arr[2] in gridZ:
        gridZ[arr[2]].append(arr[6])
    else:
        gridZ[arr[2]] = [arr[6]]

    if arr[3] in blockX:
        blockX[arr[3]].append(arr[6])
    else:
        blockX[arr[3]] = [arr[6]]

    if arr[4] in blockY:
        blockY[arr[4]].append(arr[6])
    else:
        blockY[arr[4]] = [arr[6]]

    if arr[5] in blockZ:
        blockZ[arr[5]].append(arr[6])
    else:
        blockZ[arr[5]] = [arr[6]]


for key in gridX:
    gridX[key] = sum(gridX[key])/len(gridX[key])

for key in gridY:
    gridY[key] = sum(gridY[key])/len(gridY[key])

for key in gridZ:
    gridZ[key] = sum(gridZ[key])/len(gridZ[key])

for key in blockX:
    blockX[key] = sum(blockX[key])/len(blockX[key])

for key in blockY:
    blockY[key] = sum(blockY[key])/len(blockY[key])

for key in blockZ:
    blockZ[key] = sum(blockZ[key])/len(blockZ[key])


lists1 = sorted(gridX.items())
x1, y1 = zip(*lists1)

plt.plot(x1, y1, color = 'r')
plt.title('gridX vs Time')
plt.xlabel('gridX value')
plt.ylabel('Time(s)')
plt.savefig("gridX", dpi='figure', format=None)
plt.clf()

lists2 = sorted(gridY.items())
x2, y2 = zip(*lists2)

plt.plot(x2, y2, color = 'c')
plt.title('gridY vs Time')
plt.xlabel('gridY value')
plt.ylabel('Time(s)')
plt.savefig("gridY", dpi='figure', format=None)
plt.clf()

lists3 = sorted(gridZ.items())
x3, y3 = zip(*lists3)

plt.plot(x3, y3, color = 'm')
plt.title('gridZ vs Time')
plt.xlabel('gridZ value')
plt.ylabel('Time(s)')
plt.savefig("gridZ", dpi='figure', format=None)
plt.clf()

lists4 = sorted(blockX.items())
x4, y4 = zip(*lists4)

plt.plot(x4, y4, color = 'y')
plt.title('blockX vs Time')
plt.xlabel('blockX value')
plt.ylabel('Time(s)')
plt.savefig("blockX", dpi='figure', format=None)
plt.clf()

lists5 = sorted(blockY.items())
x5, y5 = zip(*lists5)

plt.plot(x5, y5, color = 'g')
plt.title('blockY vs Time')
plt.xlabel('blockY value')
plt.ylabel('Time(s)')
plt.savefig("blockY", dpi='figure', format=None)
plt.clf()

lists6 = sorted(blockZ.items())
x6, y6 = zip(*lists6)

plt.plot(x6, y6, color = 'b')
plt.title('blockZ vs Time')
plt.xlabel('blockZ value')
plt.ylabel('Time(s)')
plt.savefig("blockZ", dpi='figure', format=None)
plt.clf()


plt.plot(x1, y1, color = 'r', label = "gridX")
plt.plot(x2, y2, color = 'c', label = "gridY")
plt.plot(x3, y3, color = 'm', label = "gridZ")
plt.plot(x4, y4, color = 'y', label = "blockX")
plt.plot(x5, y5, color = 'g', label = "blockY")
plt.plot(x6, y6, color = 'b', label = "blockZ")
plt.title('params vs Time')
plt.xlabel('param values')
plt.ylabel('Time(s)')
leg = plt.legend(loc='upper right')
plt.savefig("all", dpi='figure', format=None)
