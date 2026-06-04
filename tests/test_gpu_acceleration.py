"""
Unit tests for GPU acceleration modules.

Tests MPS availability, feature extraction accuracy, and performance.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch", reason="PyTorch not installed")

from src.utils.apple_silicon_gpu import AppleSiliconGPU
from src.analysis.gpu_features import GPUFeatureExtractor
from src.analysis.gpu_correlation import GPUCorrelation


@pytest.fixture
def gpu():
    """Initialize GPU for testing."""
    gpu = AppleSiliconGPU()
    if not gpu.use_mps:
        pytest.skip("MPS not available on this device")
    
    return gpu


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 44100
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    return audio, sr


class TestAppleSiliconGPU:
    """Test GPU detection and initialization."""
    
    def test_hardware_detection(self, gpu):
        """Test that hardware is correctly detected."""
        assert gpu.chip_model in ['M1', 'M1 Pro', 'M1 Max', 'M1 Ultra', 
                                   'M2', 'M2 Pro', 'M2 Max', 'M2 Ultra',
                                   'M3', 'M3 Pro', 'M3 Max', 'M4']
        assert gpu.gpu_cores > 0
        assert gpu.memory_gb > 0
    
    def test_mps_available(self, gpu):
        """Test that MPS backend is available."""
        assert gpu.use_mps is True
        assert gpu.device == 'mps'
    
    def test_batch_size(self, gpu):
        """Test that batch size is reasonable."""
        assert 2 <= gpu.batch_size <= 16
    
    def test_tensor_conversion(self, gpu, sample_audio):
        """Test CPU↔GPU tensor conversion."""
        audio, _ = sample_audio
        
        # Convert to GPU
        gpu_tensor = gpu.to_device(audio)
        assert gpu_tensor.device.type == 'mps'
        assert gpu_tensor.shape == audio.shape
        
        # Convert back to CPU
        cpu_array = gpu.to_numpy(gpu_tensor)
        assert isinstance(cpu_array, np.ndarray)
        np.testing.assert_allclose(cpu_array, audio, rtol=1e-5)
    
    def test_memory_management(self, gpu):
        """Test memory cache clearing."""
        # Should not raise exception
        gpu.empty_cache()
        
        memory_info = gpu.get_memory_info()
        assert 'total_gb' in memory_info
        assert 'available_gb' in memory_info


class TestGPUFeatureExtractor:
    """Test GPU-accelerated feature extraction."""
    
    def test_stft_computation(self, gpu, sample_audio):
        """Test GPU STFT produces sensible output."""
        audio, sr = sample_audio
        extractor = GPUFeatureExtractor(gpu, sr)
        
        # Compute STFT on GPU
        S_gpu = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
        
        # Verify expected shape and range
        assert S_gpu.shape[0] == 1025  # n_fft // 2 + 1 frequency bins
        assert S_gpu.shape[1] > 0      # at least one time frame
        assert np.all(np.isfinite(S_gpu))
        assert np.all(S_gpu >= 0)
        # Verify it's non-trivial (not all zeros)
        assert float(np.max(S_gpu)) > 0.01
    
    def test_chroma_computation(self, gpu, sample_audio):
        """Test GPU chroma features."""
        audio, sr = sample_audio
        extractor = GPUFeatureExtractor(gpu, sr)
        
        # Compute STFT first
        S = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
        
        # Compute chroma
        chroma = extractor.compute_chroma(S, sr)
        
        # Check shape and range
        assert chroma.shape[0] == 12  # 12 pitch classes
        assert np.all(chroma >= 0)
        assert np.all(chroma <= 1)
    
    def test_batch_processing(self, gpu, sample_audio):
        """Test batch STFT computation."""
        audio, sr = sample_audio
        extractor = GPUFeatureExtractor(gpu, sr)
        
        # Create batch of 3 identical audio samples
        batch = [audio, audio, audio]
        
        # Compute batch STFT
        results = extractor.batch_compute_stft(batch, n_fft=2048, hop_length=512)
        
        assert len(results) == 3
        for S in results:
            assert S.shape[0] > 0  # Has frequency bins
            assert S.shape[1] > 0  # Has time frames


class TestGPUCorrelation:
    """Test GPU-accelerated correlation."""
    
    def test_phase_correlation(self, gpu, sample_audio):
        """Test GPU phase correlation."""
        audio, sr = sample_audio
        correlator = GPUCorrelation(gpu)
        
        # Split audio into two segments
        mid = len(audio) // 2
        audio1 = audio[:mid]
        audio2 = audio[mid:]
        
        # Compute correlation
        correlation, offset = correlator.phase_correlation(
            audio1, audio2, window_samples=sr//2, sr=sr
        )
        
        # Check results
        assert isinstance(correlation, np.ndarray)
        assert isinstance(offset, (int, np.integer))
        assert len(correlation) > 0
    
    def test_correlation_peak(self, gpu):
        """Test that correlation finds correct alignment."""
        sr = 44100
        correlator = GPUCorrelation(gpu)
        
        # Create two identical signals with known offset.
        # Use noise (non-periodic) to avoid periodic ambiguity.
        rng = np.random.RandomState(42)
        signal = rng.randn(sr).astype(np.float32)
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        
        offset_samples = 1000
        audio1 = signal[:-offset_samples]
        audio2 = signal[offset_samples:]
        
        # Should find the offset
        _, detected_offset = correlator.phase_correlation(
            audio1, audio2, window_samples=sr//4, sr=sr
        )
        
        assert abs(detected_offset - offset_samples) < 100
    
    def test_cpu_fallback(self, sample_audio):
        """Test CPU fallback when GPU fails."""
        audio, sr = sample_audio
        
        # Create correlator without GPU
        correlator = GPUCorrelation(None)
        
        mid = len(audio) // 2
        audio1 = audio[:mid]
        audio2 = audio[mid:]
        
        # Should use CPU fallback
        correlation, offset = correlator._phase_correlation_cpu(
            audio1, audio2, window_samples=sr//2, sr=sr
        )
        
        assert isinstance(correlation, np.ndarray)
        assert isinstance(offset, (int, np.integer))


class TestPerformance:
    """Test GPU performance improvements."""
    
    def test_stft_speedup(self, gpu, sample_audio):
        """Benchmark GPU vs CPU STFT (informational, no strict assertion
        since 2s audio is too short for GPU to overcome MPS overhead)."""
        import time
        import librosa
        
        audio, sr = sample_audio
        extractor = GPUFeatureExtractor(gpu, sr)
        
        # Warm up
        _ = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
        
        # Time GPU
        start = time.perf_counter()
        for _ in range(10):
            _ = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
        gpu_time = time.perf_counter() - start
        
        # Time CPU
        start = time.perf_counter()
        for _ in range(10):
            _ = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        cpu_time = time.perf_counter() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nSTFT Speedup: {speedup:.2f}x " +
              ("✓" if speedup > 1.0 else "(GPU overhead > compute for 2s audio)"))


def test_integration():
    """Test full integration of GPU modules."""
    gpu = AppleSiliconGPU()
    if not gpu.use_mps:
        pytest.skip("MPS not available")
    
    # Create test audio
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Test feature extraction
    extractor = GPUFeatureExtractor(gpu, sr)
    S = extractor.compute_stft(audio, n_fft=2048, hop_length=512)
    chroma = extractor.compute_chroma(S, sr)
    
    assert S.shape[0] > 0
    assert chroma.shape[0] == 12
    
    # Test correlation
    correlator = GPUCorrelation(gpu)
    mid = len(audio) // 2
    correlation, offset = correlator.phase_correlation(
        audio[:mid], audio[mid:], window_samples=sr//2, sr=sr
    )
    
    assert len(correlation) > 0
    print(f"\n✓ GPU integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
