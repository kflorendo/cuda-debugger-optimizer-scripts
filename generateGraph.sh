# plot points in file input.txt
while getopts f:l: flag
do
    case "${flag}" in
        f) fileName=${OPTARG};;
        l) lineTitle=${OPTARG};;
    esac
done
echo "Input File: $fileName";
echo "Line Title: $lineTitle";


cat $fileName
gnuplot -p -e "plot '${fileName}.txt' with lines title '${lineTitle}' "
