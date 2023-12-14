DATASET="brats2018"

### pretrain

declare -a METHODS=("SimCLR" "SimSiam" "MoCo" "BYOL" "SimMIM")
for METHOD in "${METHODS[@]}"
do
    python ./pretrain.py --name "$DATASET-$METHOD" --config "configs/$DATASET/gpu.yaml" --method "$METHOD"
done


### train

declare -a METHODS=("SimCLR" "SimSiam" "MoCo" "BYOL" "SimMIM")
for METHOD in "${METHODS[@]}"
do
    python ./train.py --name "$DATASET-$METHOD" --config "configs/$DATASET/gpu.yaml" --data "./data/$DATASET/" --weights "./middle/pretrain/models/$DATASET-$METHOD/last.pth"
done


### test

declare -a METHODS=("SimCLR" "SimSiam" "MoCo" "BYOL" "SimMIM")
for METHOD in "${METHODS[@]}"
do
    python ./test.py --config "./configs/$DATASET/gpu.yaml" --model_path "./middle/models/$DATASET-$METHOD/best.pth"
    python ./post_process.py --config "./configs/$DATASET/gpu.yaml" --pred_dir "./middle/test/$DATASET-$METHOD/preds/"
done
