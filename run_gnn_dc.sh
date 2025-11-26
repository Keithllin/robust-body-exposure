# python3 assistive-gym-fem/assistive_gym/gnn_dc.py --num_seeds 50 --rollouts 35
# python3 assistive-gym-fem/assistive_gym/gnn_dc.py --num_seeds 100 --rollouts 100
# python3 assistive-gym-fem/assistive_gym/gnn_dc.py --num_seeds 100 --rollouts 100


python3 code/gen_images.py --arg_model "tl4_50_states_1k_1000_epochs=250_batch=100_workers=4_1699017034"
# python3 code/train_gnns.py --num_seeds 25
# python3 code/train_gnns.py --num_seeds 50
# python3 code/run_robe_sim.py --model-path 'tl4_50_states_1k_1000_epochs=250_batch=100_workers=4_1699017034' --graph-config 2D --env-var standard --num-rollouts 100
# python3 code/train_gnns.py --num_seeds 100
