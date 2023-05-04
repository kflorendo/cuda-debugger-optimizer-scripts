# cuda-debugger-optimizer-scripts

## Overview

Bash scripts for 15418 final project, CUDA Debugger and Optimizer (https://github.com/kflorendo/cuda-debugger-optimizer).

## Thread Output

The script `threadOutput.sh` outputs the value of a variable or expression observed by each CUDA thread.

An example of running the script is
```
./threadOutput.sh \
    -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" \
    -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" \
    -v "device_data[i+twod1-1]" \
    -t "int" \
    -l 63
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to debug.
* The `-v` flag specifies the variable or expression to get the output for.
* The `-t` flag specifies the type of the variable/expression (int, string, and float are supported).
* The `-l` flag specifies the line number to check the output of the variable/expression.

## Thread Overwriting

```
./threadOverwrite.sh \
    -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" \
    -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" \
    -v "device_data[i+twod1-1]" \
    -l 63
    -t "int" \
    -a "y"
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to debug.
* The `-v` flag specifies the variable or expression to get the check overwriting for.
* The `-l` flag specifies the line number to check overwriting of the variable/expression at.
* The `-t` flag specifies the type of the variable/expression (int, string, and float are supported).
* The `-a` flag specifies if the value is part of an array or not (and whether we should record the array index in the output).

## Time Bottleneck (GProf)

```
./timeBottleneck.sh \
    -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" \
    -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" 
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to run.
* Note: -pg flag must be added to execution command (ex. Makefile)

## Time Bottleneck (CPU/API trace)

```
./timeBottleneckTrace.sh \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" 
```

### Script Arguments

* The `-r` flag specifies the command to run the program's executable.

## Memory Bottleneck (nvprof)

```
./memoryBottleneck.sh \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" 
```

### Script Arguments

* The `-r` flag specifies the command to run the program's executable.

## Optimize Config

```
./optimizeConfig.sh \
    -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" \
    -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" \
    -b "BLOCK_DIM_X,,," \
    -g "GRID_DIM_X,,," \
    -v "2 0 0,4 0 0,8 0 0"
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to debug.
* The `-b` flag specifies the names of the macros for blockDim.x, blockDim.y, and blockDim.z, respectively (can be empty if the user doesn't want to test out permutations of this value).
* The `-g` flag specifies the names of the macros for gridDim.x, gridDim.y, and gridDim.z, respectively (can be empty if the user doesn't want to test out permutations of this value).
* The `-v` flag specifies all combinations of values to try for these block and grid dimensions (if the corresponding dimension was left empty in the `-b` and `-g`).

## Param Graph Generation

```
./generateGraph.sh \
    -f "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/output/optimizeConfig.txt"
```

### Script Arguments

* The `-f` flag specifies the path to the .txt file containing the macro values and runtime for various configurations of a program.

## Thread Breakpoint

```
./breakpoint.sh \
    -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" \
    -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" \
    -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" \
    -l 63
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to run.
* The `-l` flag specifies the line number that needs to be reached in order for a thread to be recorded as present

## Speedup Calculation

```
./speedup.sh \
    -s "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/test/threadOverwriteTest1/sequential1" \
    -p "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/test/threadOverwriteTest1/test1"
```

### Script Arguments

* The `-s` flag specifies the command to run the sequential program's executable.
* The `-p` flag specifies the command to run the parallel program's executable.
