rm -r results/nmmo
python -m torchbeast.monobeast \
    --total_steps 10000000000 \
    --learning_rate 0.001 \
    --entropy_cost 0.001 \
    --num_actors 8 \
    --num_learner 1 \
    --batch_size 32 \
    --unroll_length 32 \
    --savedir ./results \
    --checkpoint_interval 3600 \
    --xpid nmmo
