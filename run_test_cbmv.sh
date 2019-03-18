#!/bin/bash
#modelname="./models/saved/modelall_4m.rf"  # the trained random forest model, with version 0.18.2 sklearn.
modelname="./models/saved/model-1-T48-L700-entropy-Img10-CensW11-NccW3-SadW5-SobW5"  # the trained random forest model, with version 0.19.1 sklearn.
#--------------------------------
# for local expansion;
data_path="./datasets/MiddEval3/trainingQ/"
test_set="Adirondack"
resultDir="./results/localExp-cbmv/"

if [ ! -d $resultDir ]; then
		mkdir $resultDir
			echo "mkdir $resultDir"
fi

limg="${test_set}/im0.png"
rimg="${test_set}/im1.png"
disp_save="${resultDir}${test_set}_disp.pfm"
python main.py --saveCostVolume --loadCostVolume --data_path=$data_path \
 	--test_set=$test_set --model=$modelname --prob_save_path=$resultDir \
	--disp_save_path=$disp_save --l=$limg --r=$rimg --isLocalExp
