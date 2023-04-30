#!/bin/bash

echo "hello world"

while getopts m:r:c:b:g:v: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
        b) blockDimInputString=${OPTARG};;
        g) gridDimInputString=${OPTARG};;
        v) configValString=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath";

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# set comma as delimiter
IFS=','

read -a blockDimInput <<< "${blockDimInputString}"
read -a gridDimInput <<< "${gridDimInputString}"
read -a configVals <<< "${configValString}"

if [ "${#blockDimInput[@]}" -ne "3" ]; then
	echo "block dim requires 3 comma separated arguments, ${#blockDimInput[@]} provided"
fi

if [ "${#gridDimInput[@]}" -ne "3" ]; then
        echo "grid dim requires 3 comma separated arguments, ${#gridDimInput[@]} provided"
fi

echo ${blockDimInput[0]}
echo ${blockDimInput[1]}
echo ${blockDimInput[2]}

blockDimXVar=${blockDimInput[0]}
blockDimYVar=${blockDimInput[1]}
blockDimZVar=${blockDimInput[2]}

gridDimXVar=${gridDimInput[0]}
gridDimYVar=${gridDimInput[1]}
gridDimZVar=${gridDimInput[2]}

blockDimXSet=true
blockDimYSet=true
blockDimZSet=true

gridDimXSet=true
gridDimYSet=true
gridDimZSet=true

if [ "$blockDimXVar" == "" ]; then
	blockDimXSet=false
fi
if [ "$blockDimYVar" == "" ]; then
        blockDimYSet=false
fi
if [ "$blockDimZVar" == "" ]; then
        blockDimZSet=false
fi
if [ "$gridDimXVar" == "" ]; then
        gridDimXSet=false
fi
if [ "$gridDimYVar" == "" ]; then
        gridDimYSet=false
fi
if [ "$gridDimZVar" == "" ]; then
        gridDimZSet=false
fi

echo $blockDimXSet
echo $blockDimYSet
echo $blockDimZSet
echo $gridDimXSet
echo $gridDimYSet
echo $gridDimZSet

blockDimX=2048
IFS=' '
for configVal in "${configVals[@]}"
do
	echo "config val: "
	echo $configVal
	read -a config <<< "${configVal}"
	for c in "${config[@]}"
	do
		echo "c: "
		echo $c
	done
done
sed -i -E "s/#define +${blockDimInput[0]}.*/#define ${blockDimInput[0]} ${blockDimX}/" $cuPath

# copy back original contents
cp tempCu $cuPath
rm tempCu
