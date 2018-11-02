for i in {1..15}
do
    python examples/ppo.py  --name "modelv$i" --seed $RANDOM &
done