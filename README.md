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
```

### Script Arguments

* The `-m` flag specifies the absolute path to the directory containing the Makefile used to compile the program.
* The `-r` flag specifies the command to run the program's executable.
* The `-c` flag specifies the path to the .cu file to debug.
* The `-v` flag specifies the variable or expression to get the check overwriting for.
* The `-l` flag specifies the line number to check overwriting of the variable/expression at.
