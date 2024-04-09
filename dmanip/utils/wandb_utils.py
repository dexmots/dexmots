import wandb
import yaml


def delete_last_run(run_id):
    api = wandb.Api()
    if run_id == "last":
        run = api.runs(path="krshna/dmanip", per_page=1)[0]
    else:
        run = api.run(path=f"krshna/dmanip/{run_id}")
    try:
        run.delete()
    except Exception:
        return False
    else:
        print(f"deleted run: {run.id}")
        return True


def update_config(run_id, config_path):
    api = wandb.Api()
    config = yaml.safe_load(open(config_path, "r"))["params"]
    run = api.run(path=f"krshna/dmanip/{run_id}")
    run.config = config
    run.update()
