while getopts f: flag
do
    case "${flag}" in
        f) fileName=${OPTARG};;
    esac
done
echo "Input File: $fileName";

python3 config.py ${fileName}
