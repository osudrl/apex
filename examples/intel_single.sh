cd apex
visdom -port=$((6000+$1)) &
python examples/distributed_td3.py --name "modelv$1" --seed=$1 --num_actors=24