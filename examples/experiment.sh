for i in {1..10}
do
    python examples/ppo.py  --name "model$i" --seed $RANDOM &
done