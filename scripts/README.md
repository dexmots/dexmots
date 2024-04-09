Running `sim_stiff_joint.py`

```{console}
# for optimizing fewer than 1 rotation
$ python sim_stiff_joint.py --mu-static .5 --mu-dynamic .5 --duration 1. --action -1.0 --target-kd 0. --max-rotations 0.5 --optimize --lr 5e-3 --plot --save-plot
# for optimizing target theta < 1 rotation, but allowed more than 2 rotations
$ python sim_stiff_joint.py --mu-static .5 --mu-dynamic .5 --duration 5. --action -1.0 --target-kd 0. --max-rotations 1.25 --save-log --optimize --lr 1e-2
# for optimizing target theta and max rotations > 1 rotation
$ python sim_stiff_joint.py --plot --optimize --num-iter 200 --max-rotations 2 --target-theta 2.05 --duration 3 --lr .5 --save-traj --grad-clip 1 --run-id test-rotation-2 --action=1. --optimize-steps 60
```

Check env runs properly
```{console}
$ python run_env.py -p random -ne 2048 -neps 3
$ python run_env.py -p random -ne 32 -neps 3 --diff_sim
$ python compute_grad.py --num_envs 2 --num_steps 32 --save_fig
```
