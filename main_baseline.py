"""
main_baseline.py — Pure Q-learning baseline (no human feedback).

A stripped-down copy of main_oracle.py with all feedback machinery removed.
The agent learns purely from environment reward — no oracle, no Ce estimation,
no feedback injection.  Use this as the no-feedback comparison condition.

Usage
-----
  python main_baseline.py
"""

import numpy as np
from trainer import PacmanTrainer
from RLmon import RLmon
import pickle


def main(
    algID          = "tabQL_Cest_vi_t2",
    env_size       = "small",
    trial_count    = 1,
    episode_count  = 1000,
    max_steps      = 500,
    simInfo        = "_baseline",
    reset_env_random = False,
):
    print(f"start -- {algID} {simInfo}  [baseline / no feedback]")

    monitor = RLmon(["return"])

    for k in range(trial_count):
        print(f"trial: {k}")

        trainer = PacmanTrainer(
            algID=algID,
            env_size=env_size,
        )

        totalRW_list = []
        empty_fb = [[]]   # no feedback every step

        for i in range(episode_count):
            trainer.reset_episode(random=reset_env_random)

            for j in range(max_steps):
                action_idx, ob, rw, done = trainer.step(
                    feedback=empty_fb, update_Cest=False
                )

                if done or j == max_steps - 1:
                    # Terminal update — still no feedback
                    trainer.agent.act(
                        action_idx, ob, rw, done,
                        empty_fb, 0.5, update_Cest=False
                    )
                    trainer.agent.prev_obs = None
                    break

            totalRW_list.append(trainer.totRW)
            if i % 20 == 0:
                print(f"{k}, {i}: total reward mean: {np.mean(totalRW_list):+.1f}")
                totalRW_list = []

            monitor.store("return", trainer.totRW)

        monitor.store(done=True)

    fname = (
        f"results/results_Env_{env_size}_{algID}{simInfo}"
    )
    monitor.saveData(fname)
    print(f"Results saved → {fname}")


if __name__ == "__main__":
    main()
