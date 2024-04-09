diffrl manipulation
===================


## Setup

```
$ git clone --recursive git@github.com:krishpop/diff_manip.git && cd diff_manip
$ conda env create -f environment.yaml
$ cd external/
$ ./setup_libs.sh  # to install warp and external deps if you haven't already 
```

Scripts to run env/compute grads:
```{console}
$ cd scripts
$ python run_env.py -p random -ne 2048 -neps 3
$ python run_env.py -p random -ne 32 -neps 3 --diff_sim
$ python compute_grad.py --num_envs 2 --num_steps 32 --save_fig
```

To train SHAC
```{console}
$ cd external/shac/examples
$ python train_shac.py --cfg cfg/shac/claw.yaml --logdir logs/test/shac
```

To train PPO
```{console}
$ cd external/shac/examples
$ python train_rl.py --cfg cfg/ppo/claw.yaml --logdir logs/test/ppo
```
