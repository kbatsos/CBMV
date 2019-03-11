#!/bin/bash
#echo "Hi, I'm sleeping for 1 seconds..."
#sleep 1s 
#echo "all Done."
#home="/home/ccj/CBMV"
modelname="./models/modelall_4m.rf"
#--------------------------------
# for local expansion;
# for training
data_path="./datasets/MiddEval3/trainingH/"
test_set="Adirondack"
resultDir="./results/localExp-cbmv/"

if [ ! -d $resultDir ]; then
		mkdir $resultDir
			echo "mkdir $resultDir"
fi

limg="${test_set}/im0.png"
#rimg="${data_path}$test_set/im1.png"
rimg="${test_set}/im1.png"
disp_save="${resultDir}${test_set}_disp.pfm" 
python main.py --data_path=$data_path --test_set=$test_set --model=$modelname --prob_save_path=$resultDir \
	--disp_save_path=$disp_save --l=$limg --r=$rimg --isCost
