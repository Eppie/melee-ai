from __future__ import annotations
import multiprocessing as mp
import time
import torch
import os
import argparse
from pathlib import Path
from typing import Dict

from new_ppo.ppo_types import PPOConfig, InferenceRequest, InferenceResponse
from new_ppo.worker import run_worker
from new_ppo.inference import InferenceServer
from new_ppo.learner import PPOTrainer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dolphin-path", type=str, required=True)
    parser.add_argument("--iso-path", type=str, required=True)
    parser.add_argument("--opponent-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--envs-per-worker", type=int, default=8)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Multiprocessing setup
    mp.set_start_method("spawn", force=True)
    
    # Configuration
    config = PPOConfig(
        dolphin_path=args.dolphin_path,
        iso_path=args.iso_path,
        opponent_path=args.opponent_path,
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        rollout_length=args.rollout_length,
        total_timesteps=args.total_timesteps,
        device=args.device if torch.cuda.is_available() else "cpu"
    )
    
    print("=== Starting Batched PPO Training ===")
    print(f"Workers: {config.num_workers}")
    print(f"Envs/Worker: {config.envs_per_worker}")
    print(f"Total Envs: {config.num_workers * config.envs_per_worker}")
    print(f"Device: {config.device}")
    
    # Queues
    state_queue: mp.Queue[InferenceRequest] = mp.Queue() # All workers write to this
    action_queues: Dict[int, mp.Queue[InferenceResponse]] = {i: mp.Queue() for i in range(config.num_workers)} # Server writes to specific worker
    
    # Spawn Workers
    workers = []
    for i in range(config.num_workers):
        p = mp.Process(
            target=run_worker,
            args=(i, config, state_queue, action_queues[i]),
            name=f"Worker-{i}"
        )
        p.start()
        workers.append(p)
        
    try:
        # Init Server & Trainer
        server = InferenceServer(config, state_queue, action_queues)
        trainer = PPOTrainer(config, server.learner)
        
        total_steps = 0
        epoch = 0
        
        while total_steps < config.total_timesteps:
            epoch += 1
            print(f"\n--- Epoch {epoch} ---")
            
            # 1. Collect Data
            server.collect_rollout()
            
            # 2. Train
            trainer.train_epoch(server.buffer)
            
            # Metrics
            steps_in_epoch = config.rollout_length * config.num_workers * config.envs_per_worker
            total_steps += steps_in_epoch
            
            print(f"Total Steps: {total_steps}")
            
            # Save Checkpoint
            if epoch % 10 == 0:
                path = f"checkpoints/new_ppo_step_{total_steps}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'model': server.learner.state_dict(),
                    'config': config
                }, path)
                print(f"Saved checkpoint to {path}")
                
    except KeyboardInterrupt:
        print("\nStopping training...")
    finally:
        # Cleanup
        print("Shutting down workers...")
        for p in workers:
            p.terminate() # Aggressive termination
            p.join()
            
if __name__ == "__main__":
    main()
