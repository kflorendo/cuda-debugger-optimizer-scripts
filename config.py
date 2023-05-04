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
    arr = cleanline.split()

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


lists = sorted(gridX.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'r')
plt.title('gridX vs Time')
plt.savefig("gridX", dpi='figure', format=None)
plt.clf()

lists = sorted(gridY.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'c')
plt.title('gridY vs Time')
plt.savefig("gridY", dpi='figure', format=None)
plt.clf()

lists = sorted(gridZ.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'm')
plt.title('gridZ vs Time')
plt.savefig("gridZ", dpi='figure', format=None)
plt.clf()

lists = sorted(blockX.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'y')
plt.title('blockX vs Time')
plt.savefig("blockX", dpi='figure', format=None)
plt.clf()

lists = sorted(blockY.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'g')
plt.title('blockY vs Time')
plt.savefig("blockY", dpi='figure', format=None)
plt.clf()

lists = sorted(blockZ.items())
x, y = zip(*lists)

plt.plot(x, y, color = 'b')
plt.title('blockZ vs Time')
plt.savefig("blockZ", dpi='figure', format=None)
