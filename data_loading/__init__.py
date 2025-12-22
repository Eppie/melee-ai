"""Data loading utilities and configuration.

This module sets up global configuration for data loading operations.
"""
import os

# Optimize Zarr/Blosc decompression for multi-process dataloading
# Prevents thread oversubscription when using multiple workers
# This MUST be set before any Blosc operations occur
os.environ.setdefault("BLOSC_NTHREADS", "1")
