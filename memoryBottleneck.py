import matplotlib.pyplot as plt

file1 = open(f'output/timeBottleneckGpuTrace.txt', 'r')
lines = file1.readlines()

num = 0
catSet = set()
data = []

for line in lines:
    if num >= 5:
        lineArr = line.split(',')
        start = float(lineArr[0])
        staticMem = float(lineArr[9])
        dynamicMem = float(lineArr[10])
        size = float(lineArr[11])
        throughput = float(lineArr[12])
        nameArr = lineArr[18:-1]
        name = ','.join(nameArr)
        data.append((start, staticMem, dynamicMem, size, throughput, name))
    num += 1

#TODO: graph
