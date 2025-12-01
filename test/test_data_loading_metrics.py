"""Unit tests for DataLoadingMetrics."""

from __future__ import annotations

import time

import pytest

from data_loading.instrumentation import DataLoadingMetrics


class TestDataLoadingMetrics:
    """Test DataLoadingMetrics functionality."""

    def test_initialization(self):
        """Test DataLoadingMetrics initialization."""
        metrics = DataLoadingMetrics()

        assert len(metrics.chunk_load_times) == 0
        assert len(metrics.chunk_load_iterations) == 0
        assert len(metrics.batch_prep_times) == 0
        assert len(metrics.gpu_idle_times) == 0
        assert metrics.last_batch_end_time == 0.0

    def test_record_chunk_load(self):
        """Test recording chunk load times."""
        metrics = DataLoadingMetrics()

        metrics.record_chunk_load(2.5, iteration=0)
        metrics.record_chunk_load(3.0, iteration=1)
        metrics.record_chunk_load(2.8, iteration=2)

        assert len(metrics.chunk_load_times) == 3
        assert metrics.chunk_load_times == [2.5, 3.0, 2.8]
        assert metrics.chunk_load_iterations == [0, 1, 2]

    def test_record_batch_prep(self):
        """Test recording batch preparation times."""
        metrics = DataLoadingMetrics()

        for i in range(5):
            metrics.record_batch_prep(0.01 * (i + 1))

        assert len(metrics.batch_prep_times) == 5
        assert metrics.batch_prep_times[-1] == 0.05

    def test_batch_prep_rolling_window(self):
        """Test that batch_prep_times maintains rolling window of 100."""
        metrics = DataLoadingMetrics()

        # Add 150 entries
        for i in range(150):
            metrics.record_batch_prep(0.01)

        # Should keep only last 100
        assert len(metrics.batch_prep_times) == 100

    def test_record_batch_start_and_end(self):
        """Test batch start/end recording for idle time calculation."""
        metrics = DataLoadingMetrics()

        # First batch - no idle time yet
        metrics.record_batch_start()
        time.sleep(0.01)  # Simulate training
        metrics.record_batch_end()

        assert len(metrics.gpu_idle_times) == 0  # First batch has no idle

        # Second batch - should calculate idle time
        time.sleep(0.02)  # Simulate gap between batches
        metrics.record_batch_start()
        time.sleep(0.01)  # Simulate training
        metrics.record_batch_end()

        # Should have recorded one idle time
        assert len(metrics.gpu_idle_times) == 1
        assert metrics.gpu_idle_times[0] > 0.015  # At least the sleep time

    def test_iteration_duration_tracking(self):
        """Test iteration duration calculation."""
        metrics = DataLoadingMetrics()

        # Record several iterations
        for _ in range(3):
            metrics.record_batch_start()
            time.sleep(0.01)
            metrics.record_batch_end()

        # Should have 3 iteration durations (after first completes)
        assert len(metrics.iteration_durations) == 3
        for duration in metrics.iteration_durations:
            assert duration >= 0.01

    def test_get_summary_empty(self):
        """Test get_summary with no data."""
        metrics = DataLoadingMetrics()

        summary = metrics.get_summary()

        # Should return empty dict or dict with no metrics
        assert isinstance(summary, dict)
        assert len(summary) == 0

    def test_get_summary_chunk_load_metrics(self):
        """Test summary statistics for chunk loading."""
        metrics = DataLoadingMetrics()

        metrics.record_chunk_load(2.0, iteration=0)
        metrics.record_chunk_load(3.0, iteration=1)
        metrics.record_chunk_load(2.5, iteration=2)

        summary = metrics.get_summary()

        assert "dataloader/chunk_load_mean_s" in summary
        assert summary["dataloader/chunk_load_mean_s"] == 2.5
        assert summary["dataloader/chunk_load_max_s"] == 3.0
        assert summary["dataloader/chunk_load_min_s"] == 2.0
        assert summary["dataloader/chunk_load_total_s"] == 7.5
        assert summary["dataloader/num_chunks_loaded"] == 3

    def test_get_summary_batch_prep_metrics(self):
        """Test summary statistics for batch preparation."""
        metrics = DataLoadingMetrics()

        for i in range(10):
            metrics.record_batch_prep(0.01 * (i + 1))

        summary = metrics.get_summary()

        assert "dataloader/batch_prep_mean_ms" in summary
        assert "dataloader/batch_prep_p50_ms" in summary
        assert "dataloader/batch_prep_p95_ms" in summary
        assert "dataloader/batch_prep_max_ms" in summary

        # Check values are in milliseconds
        assert summary["dataloader/batch_prep_max_ms"] == 0.1 * 1000  # 0.1s = 100ms

    def test_get_summary_gpu_idle_metrics(self):
        """Test summary statistics for GPU idle time."""
        metrics = DataLoadingMetrics()

        # Simulate several batches with gaps
        for _ in range(10):
            metrics.record_batch_start()
            time.sleep(0.005)
            metrics.record_batch_end()
            time.sleep(0.002)  # Gap between batches

        summary = metrics.get_summary()

        assert "dataloader/gpu_idle_mean_ms" in summary
        assert "dataloader/gpu_idle_p50_ms" in summary
        assert "dataloader/gpu_idle_p95_ms" in summary
        assert "dataloader/gpu_idle_max_ms" in summary

        # Idle time should be roughly the sleep time (2ms)
        assert 1.5 < summary["dataloader/gpu_idle_mean_ms"] < 5.0

    def test_get_summary_gpu_utilization(self):
        """Test GPU utilization estimate calculation."""
        metrics = DataLoadingMetrics()

        # Simulate batches with minimal idle time (good utilization)
        for _ in range(10):
            metrics.record_batch_start()
            time.sleep(0.01)  # 10ms training
            metrics.record_batch_end()
            time.sleep(0.001)  # 1ms idle

        summary = metrics.get_summary()

        assert "dataloader/gpu_utilization_estimate" in summary
        # Utilization should be high: 10ms / (10ms + 1ms) ≈ 0.91
        assert summary["dataloader/gpu_utilization_estimate"] > 0.85

    def test_get_summary_iteration_metrics(self):
        """Test summary statistics for iteration timing."""
        metrics = DataLoadingMetrics()

        for _ in range(10):
            metrics.record_batch_start()
            time.sleep(0.01)
            metrics.record_batch_end()

        summary = metrics.get_summary()

        assert "dataloader/iteration_mean_ms" in summary
        assert "dataloader/iteration_p95_ms" in summary
        assert "dataloader/batches_per_second" in summary

        # Iterations should be ~10ms
        assert 8 < summary["dataloader/iteration_mean_ms"] < 15

        # Batches per second should be roughly 1 / 0.01 = 100
        assert 60 < summary["dataloader/batches_per_second"] < 150

    def test_reset_epoch(self):
        """Test resetting epoch-level metrics."""
        metrics = DataLoadingMetrics()

        # Record some data
        metrics.record_chunk_load(2.5, iteration=0)
        metrics.record_batch_prep(0.01)
        metrics.record_batch_start()
        metrics.record_batch_end()

        # Reset epoch-level metrics
        metrics.reset_epoch()

        # Chunk load times should be preserved
        assert len(metrics.chunk_load_times) == 1

        # Batch-level metrics should be cleared
        assert len(metrics.batch_prep_times) == 0
        assert len(metrics.gpu_idle_times) == 0
        assert len(metrics.iteration_start_times) == 0
        assert len(metrics.iteration_durations) == 0
        assert metrics.last_batch_end_time == 0.0

    def test_get_chunk_load_report_empty(self):
        """Test chunk load report with no data."""
        metrics = DataLoadingMetrics()

        report = metrics.get_chunk_load_report()

        assert "No chunks loaded yet" in report

    def test_get_chunk_load_report_with_data(self):
        """Test chunk load report with data."""
        metrics = DataLoadingMetrics()

        metrics.record_chunk_load(2.0, iteration=0)
        metrics.record_chunk_load(3.0, iteration=1)
        metrics.record_chunk_load(2.5, iteration=2)
        metrics.record_chunk_load(4.0, iteration=3)

        report = metrics.get_chunk_load_report()

        assert "Total chunks loaded: 4" in report
        assert "Mean load time: 2.88s" in report
        assert "Slowest chunks:" in report
        # Should show iteration 3 with 4.0s as slowest
        assert "Iteration 3: 4.00s" in report

    def test_concurrent_metric_recording(self):
        """Test that metrics can handle rapid recording."""
        metrics = DataLoadingMetrics()

        # Rapidly record many metrics
        for i in range(1000):
            metrics.record_chunk_load(0.1, iteration=i)
            metrics.record_batch_prep(0.001)
            metrics.record_batch_start()
            metrics.record_batch_end()

        summary = metrics.get_summary()

        # Should have all chunk loads
        assert summary["dataloader/num_chunks_loaded"] == 1000

        # Should have only last 100 batch prep times (rolling window)
        assert len(metrics.batch_prep_times) == 100

    def test_edge_case_zero_idle_time(self):
        """Test handling of zero or negative idle time."""
        metrics = DataLoadingMetrics()

        # Record batches with no gap (or negative due to timing precision)
        metrics.record_batch_start()
        metrics.record_batch_end()

        # Immediately start next batch (no gap)
        metrics.record_batch_start()

        # Should not record negative idle time
        summary = metrics.get_summary()

        if "dataloader/gpu_idle_mean_ms" in summary:
            assert summary["dataloader/gpu_idle_mean_ms"] >= 0

    def test_percentile_calculation(self):
        """Test percentile calculations in summary."""
        metrics = DataLoadingMetrics()

        # Record batch prep times: 1ms, 2ms, 3ms, ..., 100ms
        for i in range(100):
            metrics.record_batch_prep((i + 1) * 0.001)

        summary = metrics.get_summary()

        # P50 should be ~50.5ms (median of 1-100)
        assert 45 < summary["dataloader/batch_prep_p50_ms"] < 55

        # P95 should be ~95.5ms
        assert 90 < summary["dataloader/batch_prep_p95_ms"] < 100


class TestDataLoadingMetricsIntegration:
    """Integration tests simulating realistic training scenarios."""

    def test_realistic_training_loop(self):
        """Simulate a realistic training loop."""
        metrics = DataLoadingMetrics()

        # Simulate loading 4 chunks
        for chunk_idx in range(4):
            metrics.record_chunk_load(2.0 + chunk_idx * 0.5, iteration=chunk_idx)

        # Simulate 100 training iterations
        for _ in range(100):
            metrics.record_batch_start()
            # Batch prep (1-2ms)
            batch_prep_time = 0.001 + (time.time() % 0.001)
            metrics.record_batch_prep(batch_prep_time)
            # Training (10-15ms)
            time.sleep(0.01 + (time.time() % 0.005))
            metrics.record_batch_end()

        summary = metrics.get_summary()

        # Verify all metrics are present
        assert "dataloader/chunk_load_mean_s" in summary
        assert "dataloader/batch_prep_mean_ms" in summary
        assert "dataloader/gpu_idle_mean_ms" in summary
        assert "dataloader/gpu_utilization_estimate" in summary
        assert "dataloader/batches_per_second" in summary

        # Verify reasonable values
        assert summary["dataloader/num_chunks_loaded"] == 4
        assert summary["dataloader/gpu_utilization_estimate"] > 0.5

    def test_epoch_boundary_handling(self):
        """Test metrics across epoch boundaries."""
        metrics = DataLoadingMetrics()

        # Epoch 1
        for _ in range(50):
            metrics.record_batch_start()
            time.sleep(0.005)
            metrics.record_batch_end()

        epoch_1_summary = metrics.get_summary()

        # Reset for epoch 2
        metrics.reset_epoch()

        # Epoch 2
        for _ in range(50):
            metrics.record_batch_start()
            time.sleep(0.005)
            metrics.record_batch_end()

        epoch_2_summary = metrics.get_summary()

        # Summaries should be independent
        assert "dataloader/iteration_mean_ms" in epoch_2_summary

    def test_slow_chunk_identification(self):
        """Test identifying slow chunk loads."""
        metrics = DataLoadingMetrics()

        # Most chunks are fast, one is slow
        metrics.record_chunk_load(1.0, iteration=0)
        metrics.record_chunk_load(1.1, iteration=1)
        metrics.record_chunk_load(5.0, iteration=2)  # Slow!
        metrics.record_chunk_load(1.2, iteration=3)

        report = metrics.get_chunk_load_report()

        # Report should highlight the slow chunk
        assert "Iteration 2: 5.00s" in report
        assert "Max load time: 5.00s" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
