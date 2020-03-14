source ~/anaconda3/etc/profile.d/conda.sh
conda activate py37
array=( 2 15 25 )
layers=( 1 3 5 )
for lsize in "${array[@]}"
do
    for lnum in "${layers[@]}"
    do
        python rvae_training.py --hidden_size=40 --latent_size=${lsize} --num_layers=${lnum} --run_name="mj_state_SSE_KL_NoXY_latent_${lsize}_layers_${lnum}_hidden_40" &
        echo "started process mj_state_SSE_KL_NoXY_latent_${lsize}_layers_${lnum}_hidden_40"
    done
done