"""Tests for ryze.cluster.resource module."""

from ryze.cluster.resource import GPUInfo, ResourceTracker


class TestResourceTracker:
    def test_register_and_list(self):
        tracker = ResourceTracker()
        gpu = GPUInfo(gpu_id="gpu-0", name="A100", memory_total_gb=80.0)
        tracker.register_gpu(gpu)
        assert len(tracker.list_gpus()) == 1

    def test_has_capacity(self):
        tracker = ResourceTracker()
        tracker.register_gpu(GPUInfo(gpu_id="gpu-0", name="A100", memory_total_gb=80.0))
        tracker.register_gpu(GPUInfo(gpu_id="gpu-1", name="A100", memory_total_gb=80.0))
        assert tracker.has_capacity(gpu_count=2) is True
        assert tracker.has_capacity(gpu_count=3) is False

    def test_allocate_and_release(self):
        tracker = ResourceTracker()
        tracker.register_gpu(GPUInfo(gpu_id="gpu-0", name="A100", memory_total_gb=80.0))
        tracker.register_gpu(GPUInfo(gpu_id="gpu-1", name="A100", memory_total_gb=80.0))

        allocated = tracker.allocate("task-1", 1)
        assert len(allocated) == 1
        assert len(tracker.available_gpus()) == 1

        released = tracker.release("task-1")
        assert released == 1
        assert len(tracker.available_gpus()) == 2
