
threads = []

for line in open('output/threadbp.txt', 'r'):
    threads.append(line)

ordered = sorted(threads)

# clear file
with open("output/threadbp.txt", "w") as file:
    pass

with open("output/threadbp.txt", "w") as file:
    file.writelines("%s\n" % thread for thread in ordered)
