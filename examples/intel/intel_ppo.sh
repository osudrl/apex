cd apex
visdom -port=$((6000+$1)) &
python examples/ppo.py --name "ppo_modelv$1" --seed=$1 --num_actors=24 --viz_port=$((6000+$1))