"""
Hierarchical PPO training system for Melee bot.

Architecture:
- Coordinator (CRD): GPU process managing inference and PPO training
- ArenaShard (S8): CPU processes managing Dolphin instances
- EnvWorker (ENV): Threads within S8, one per Dolphin
"""

__version__ = "0.1.0"
