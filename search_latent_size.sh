source ~/anaconda3/etc/profile.d/conda.sh
conda activate py37
array=( 2 5 10 15 20 25 )
for lsize in "${array[@]}"
do
    #echo "python vae_training.py --hidden_size=40 --latent_size=${lsize} --run_name="mj_state_qpos_SSE_latent_${lsize}_hidden_40" &"
    python vae_training.py --hidden_size=40 --latent_size=${lsize} --run_name="mj_state_SSE_KL_NoXY_500epoch_latent_${lsize}_hidden_40" &
done