
threads = []

for line in open('threadbp.txt', 'r'):
    threads.append(line)

ordered = sorted(threads)