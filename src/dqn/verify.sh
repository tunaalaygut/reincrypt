# python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -s 2378
# python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -s 2378 -k 0.2
# python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -s 2378 -k 0.3
# python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -s 2378 -k 0.4
python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model
python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -k 0.2
python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -k 0.3
python3 main.py -v -i ../../data/verification -c $1 -m output/$1/$1_model -k 0.4