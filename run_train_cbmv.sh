#python main.py --train --data_path "./datasets/" --train_add ./mb/additional.txt --train_set ./mb/trainall.txt --model ./models/model.rf
n_jobs=10
python main.py --train --data_path "./datasets/MiddEval3/trainingQ/" --train_set ./mb/trainset_small.txt \
	--model ./models/model_small.rf --n_jobs=$n_jobs
