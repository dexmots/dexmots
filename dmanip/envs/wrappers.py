import os
import numpy as np
from dmanip.utils.common import to_numpy
from gym import Wrapper
from dmanip.envs.environment import RenderMode
from imageio import get_writer


class Monitor:
    def __init__(self, env, save_dir, ep_filter=None):
        self.env = env
        self.writer = None
        self.save_dir = save_dir or "./videos/"
        print("saving videos to", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if isinstance(ep_filter, int):  # interpret as save_frew of number of episodes
            save_freq = ep_filter
            ep_filter = lambda x: x % save_freq == 0
        self.ep_filter = ep_filter
        self.num_episodes = 0

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        # self.env.renderer.move_camera(np.zeros(3), 5, 225, -20)  # resets default camera pose
        if self.writer:
            self.writer.close()
        if self.ep_filter is None or self.ep_filter(self.num_episodes):
            self.writer = get_writer(
                os.path.join(self.save_dir, f"ep-{self.num_episodes}.mp4"), fps=int(1 / self.env.frame_dt)
            )
        else:
            self.writer = None
        self.num_episodes += 1
        return ret

    def step(self, action):
        res = self.env.step(action)
        if self.writer is not None:
            self.render()
        return res

    def render(self):
        if self.writer is None:
            return
        img = self.env.render(mode="rgb_array")
        self.writer.append_data((255 * img).astype(np.uint8))
        return

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def close(self):
        self.env.close()
        if self.writer is not None:
            self.writer.close()


class InfoLogger(Wrapper):
    def __init__(self, env, score_keys=[]):
        self.env = env
        self.log_data = []
        self.score_keys = score_keys
        self.mean_scores_map = {k + "_final": [] for k in self.score_keys}

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict) and done_indices.shape[0] > 0:
            for k, v in filter(lambda kv: kv[0] in self.score_keys, infos.items()):
                final_v = v[done_indices]
                if final_v.shape[0] > 0:
                    self.mean_scores_map[f"{k}_final"].append(to_numpy(final_v))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done_indices = done.nonzero(as_tuple=False).flatten()
        if len(done_indices) > 0:
            self.process_infos(info, done_indices)
            self.log_data.append(
                {
                    "progress_buf": to_numpy(self.env.progress_buf[done]),
                    "is_success": to_numpy(info.get("is_success")[done]),
                    "consecutive_successes": to_numpy(info.get("consecutive_successes")[done]),
                }
            )
        return obs, reward, done, info

    def close(self):
        save_dict = {}
        for k in filter(lambda k: len(self.mean_scores_map[k]) > 0, self.mean_scores_map):
            save_dict[k] = np.concatenate(self.mean_scores_map[k])

        if save_dict:
            np.savez("env_scores.npz", **save_dict)
            np.save("env_successes.npy", self.log_data)
        consecutive_successes = np.concatenate(list(map(lambda x: x["consecutive_successes"], self.log_data)))
        successes = np.concatenate(list(map(lambda x: x["is_success"], self.log_data))) | consecutive_successes > 0
        success_rate = successes.sum() / successes.size
        print("Num Episodes", successes.size)
        print("Successes / Num Episodes = ", f"{successes.sum()}/{successes.size} = {success_rate}")
