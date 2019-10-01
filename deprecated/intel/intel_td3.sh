cd apex
visdom -port=$((6000+$1)) &
python distributed_td3.py --name "td3_modelv$1" --seed=$1 --num_actors=24 --viz_port=$((6000+$1))


