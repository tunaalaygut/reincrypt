INPUT_DIR=$1
CONFIG=$2
K=$3
COMMAND="python main.py -v -i ${INPUT_DIR} -c ${CONFIG} -m output/${CONFIG}/${CONFIG}_model"

if ! [ -z "$3" ]
  then
    COMMAND="${COMMAND} -k ${K}"
fi

$COMMAND
