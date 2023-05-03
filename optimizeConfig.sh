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

rm output/optimizeConfig.txt
touch output/optimizeConfig.txt

IFS=' '
for configVal in "${configVals[@]}"
do
	echo "config val: "
	echo $configVal
	read -a config <<< "${configVal}"
	if [ "$blockDimXSet" = true ] ; then
                sed -i -E "s/#define +${blockDimXVar}.*/#define ${blockDimXVar} ${config[0]}/" $cuPath
	fi
	if [ "$blockDimYSet" = true ] ; then
                sed -i -E "s/#define +${blockDimYVar}.*/#define ${blockDimYVar} ${config[1]}/" $cuPath
	fi
	if [ "$blockDimZSet" = true ] ; then
                sed -i -E "s/#define +${blockDimZVar}.*/#define ${blockDimZVar} ${config[2]}/" $cuPath
	fi
	if [ "$gridDimXSet" = true ] ; then
                sed -i -E "s/#define +${gridDimXVar}.*/#define ${gridDimXVar} ${config[3]}/" $cuPath
	fi
	if [ "$gridDimYSet" = true ] ; then
                sed -i -E "s/#define +${gridDimYVar}.*/#define ${gridDimYVar} ${config[4]}/" $cuPath
	fi
	if [ "$gridDimZSet" = true ] ; then
                sed -i -E "s/#define +${gridDimZVar}.*/#define ${gridDimZVar} ${config[5]}/" $cuPath
	fi
	
	(cd $makeDir && make)
	for i in {1..3}
	do
		start_time=$(date +%s.%6N)
		$runCmd
		end_time=$(date +%s.%6N)
		elapsed=$(echo "scale=6; $end_time - $start_time" | bc)
		echo "${configVal} ${elapsed}" >> output/optimizeConfig.txt
	done
done

# copy back original contents
cp tempCu $cuPath
rm tempCu
