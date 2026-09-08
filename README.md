# Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People

This code accompanies the submission:  
["Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People"](https://arxiv.org/abs/2304.04822)

Kavya Puthuveetil, Sasha Wald, Atharva Pusalkar, Pratyusha Karnati, and Zackory Erickson

## Citation
##### ["Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People"](https://arxiv.org/abs/2304.04822)
K. Puthuveetil, Sasha Wald, Atharva Pusalkar, Pratyusha Karnati, and Z. Erickson, “Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People,” 2023.

```
@misc{puthuveetil2023robust,
      title={Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations}, 
      author={Kavya Puthuveetil and Sasha Wald and Atharva Pusalkar and Pratyusha Karnati and Zackory Erickson},
      year={2023},
      eprint={2304.04822}, 
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Install

### General Packages

To run the RoBE framework, you will need install the following packages according to their respective installation instructions. We have provided the package versions that we used, though other versions may also work.
1. [pytorch](https://pytorch.org/get-started/previous-versions/#v1101)==1.10.1+cu113
2. [torch_geometric](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html)==2.0.3
3. [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)==2.9.1
4. [cma](https://github.com/CMA-ES/pycma)==3.1.0
5. `scipy` and `trimesh` for radius and ground-truth mesh graph construction
6. `gradient-free-optimizers` for the optional Recover random-search baseline


### Assistive Gym

This repository provides a version of [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) modified for this work, as well as additional task-specific functionality. **Although files for other assistive environments are included, ONLY the RoBE Bedding Manipulation enviornment is functional!**

For more details on installing the version of Assistive Gym contained in this repository, check out the [installation guide for Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym/wiki/1.-Install). Just replace the lines that say `git clone https://github.com/Healthcare-Robotics/assistive-gym.git` and `cd assistive-gym` with:
```
git clone https://github.com/RCHI-Lab/robust-body-exposure.git
cd robust-body-exposure/assistive-gym-fem
```
Generating the actuated human model in the RoBE Bedding Manipulation environment relies on SMPL-X human mesh models. In order to use these models, you will need to create an account at https://smpl-x.is.tue.mpg.de/index.html and [download](https://smpl-x.is.tue.mpg.de/download.php) the mesh models. Once downloaded, extract the file and move the entire `smplx` directory to `robust-body-exposure/assistive-gym-fem/assistive_gym/envs/assets/smpl_models/`. Once complete, you should have several files with this format: `robust-body-exposure/assistive-gym-fem/assistive_gym/envs/assets/smpl_models/smplx/SMPLX_FEMALE.npz`. This step is REQUIRED to run the RoBE Bedding Manipulation environment!

## Download Models
To run RoBE with our pre-trained dynamics models, download them (15.5 GB)
from the following link: [pre-trained dynamics models](https://drive.google.com/drive/folders/1pJbTdy3lsDDvSy7WUoEhFkFN9oaKVIUX?usp=sharing).

Once downloaded, move the unzipped `trained_models` directory (with all sub-directories also unzipped) into the `robust-body-exposure` directory. The final path to the dynamics models, for example, should be `robust-body-exposure/trained_models/GNN` 


## Basics
The RoBE Bedding Manipulation environment, built in [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym), can be visualized using the following command:
```
PYTHONPATH=assistive-gym-fem python3 -m assistive_gym --env RobeReversible-v1
```

## Running RoBE in Simulation

Lets try running an evaluation of RoBE in simulation! All of the commands below assume that they are being run from the `robust-body-exposure` directory so please `cd` accordingly!

To optimize over a pre-trained RoBE dynamics model and uncover randomly selected target limbs over 100 simulation rollouts from the training distribution:
```
python3 code/run_robe_sim.py --model-path 'standard_2D_10k_epochs=250_batch=100_workers=4_1668718872' --graph-config 2D --env-var standard --num-rollouts 100
```

The same Uncover entry point exposes the entropy-weighted objective used for
the W100--W400 comparisons. For a fixed evaluation manifest, set the weight
and occupancy-grid size explicitly:
```
python3 code/run_robe_sim.py --model-path 'Uncover/<MODEL>' --graph-config 2D --env-var standard --eval-set <EVAL_SET.json> --entropy-weight 200 --entropy-grid-size 0.05
```

For Recover CMA/random-search evaluation, use the paper's sequential Recover
entry point. `--graph-config 2D` is the radius-graph contract; use `2D_mesh`
only with a checkpoint trained on the ground-truth blanket mesh:
```
python3 code/run_robe_sim_new_opt.py --models-dir trained_models/FINAL_MODELS --model-path 'Recover/<MODEL>' --graph-config 2D --env-var standard --eval-set <EVAL_SET.json> --warm-start-strategy line --search-method cma
```

## Training New Dynamics Models

Given a new dataset of cloth interactions, place raw pickle files under
`DATASETS/<DATASET-NAME>/raw`. The Recover entry point supports explicit
dataset sizes and radius/mesh graph selection:
```
python3 code/train_gnns.py --recover --dataset-dir DATASETS/<DATASET-NAME> --dataset-sizes 70000 --description <DATASET-DESCRIPTION> --model-prefix <MODEL-NAME> --edge-mode radius
```

Simulation data collection uses the two task-specific entry points:
```
python3 assistive-gym-fem/assistive_gym/gnn_dc_uncover.py --output-dataset-dir DATASETS/Uncover_Data/<RUN>
python3 assistive-gym-fem/assistive_gym/gnn_dc_recover.py --output-dataset-dir DATASETS/Recover_Data/<RUN>
```

## Real-world closed loop

The human-study production path is orchestrated by `real_world/run_trial.py`.
It captures the initial pose and blanket point cloud, plans and executes
Uncover, captures the intermediate state, plans and executes Recover, and
computes the corresponding real-world scores. Follow the portable setup
instructions in `real_world/SESSION_REGISTRATION.md` and
`real_world/ros2/README.md`; camera calibration, session artifacts, models,
and study data stay outside the repository.
```
python3 real_world/run_trial.py --help
```




