"""
Apple Silicon GPU acceleration using Metal Performance Shaders (MPS).

Provides a unified interface for GPU-accelerated audio feature extraction.
Automatically detects chip model and sets optimal batch sizes.
Falls back gracefully to CPU when MPS is unavailable.
"""

import platform
import subprocess
import logging
import warnings

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded torch — don't fail at import time if torch is missing
_torch = None
_import_attempted = False


def _get_torch():
    """Lazy-load PyTorch to avoid import errors when not installed."""
    global _torch, _import_attempted
    if not _import_attempted:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = None
        _import_attempted = True
    return _torch


class MetalGPU:
    """GPU acceleration wrapper for Apple Silicon M-series chips."""

    def __init__(self):
        self.device = 'cpu'
        self.use_mps: bool = False
        self.chip_model: Optional[str] = None
        self.gpu_cores: int = 0
        self.memory_gb: float = 0.0
        self.batch_size: int = 4

        self._detect_hardware()
        self._initialize_mps()
        self._set_optimal_batch_size()

    # ── Hardware Detection ──────────────────────────────────────────

    def _detect_hardware(self):
        """Detect Apple Silicon chip model and memory."""
        try:
            if platform.processor() != 'arm' or platform.system() != 'Darwin':
                return

            result = subprocess.check_output(
                ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
                stderr=subprocess.DEVNULL
            )
            chip_name = result.decode().strip()

            # Chip → GPU core mapping (covers M1 through M4 Ultra)
            chip_map = {
                'M1 Ultra': 64, 'M1 Max': 32, 'M1 Pro': 16, 'M1': 8,
                'M2 Ultra': 76, 'M2 Max': 38, 'M2 Pro': 19, 'M2': 10,
                'M3 Ultra': 80, 'M3 Max': 40, 'M3 Pro': 18, 'M3': 10,
                'M4 Ultra': 80, 'M4 Max': 40, 'M4 Pro': 20, 'M4': 10,
            }

            for model, cores in chip_map.items():
                if model in chip_name:
                    self.chip_model = model
                    self.gpu_cores = cores
                    break
            else:
                self.chip_model = chip_name
                self.gpu_cores = 8

            mem_bytes = subprocess.check_output(
                ['/usr/sbin/sysctl', '-n', 'hw.memsize'], stderr=subprocess.DEVNULL
            )
            self.memory_gb = int(mem_bytes.decode().strip()) / (1024 ** 3)

            logger.info(f"Apple Silicon: {self.chip_model} | "
                        f"{self.gpu_cores} GPU cores | {self.memory_gb:.0f} GB")

        except Exception as e:
            logger.debug(f"Hardware detection failed: {e}")

    def _initialize_mps(self):
        """Initialize Metal Performance Shaders backend."""
        if not self.chip_model:
            return

        torch = _get_torch()
        if torch is None:
            logger.info("PyTorch not installed — GPU acceleration disabled")
            return

        try:
            major, minor = map(int, torch.__version__.split('+')[0].split('.')[:2])
            if major < 1 or (major == 1 and minor < 12):
                logger.warning(f"PyTorch {torch.__version__} too old for MPS (need 1.12+)")
                return

            if not torch.backends.mps.is_available():
                logger.warning("MPS backend not available")
                return

            # Smoke test
            test = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            result = (test * 2).cpu()
            if torch.allclose(result, torch.tensor([2.0, 4.0, 6.0])):
                self.device = torch.device('mps')
                self.use_mps = True
                logger.info("✓ Metal Performance Shaders (MPS) initialized")
            else:
                logger.warning("MPS smoke test failed")

        except Exception as e:
            logger.warning(f"MPS initialization failed: {e}")

    def _set_optimal_batch_size(self):
        """Set batch size based on available unified memory."""
        if not self.use_mps or self.memory_gb == 0:
            return

        if self.memory_gb < 8:
            self.batch_size = 2
        elif self.memory_gb < 16:
            self.batch_size = 4
        elif self.memory_gb < 32:
            self.batch_size = 8
        elif self.memory_gb < 64:
            self.batch_size = 12
        else:
            self.batch_size = 16

    # ── Data Transfer ───────────────────────────────────────────────

    def to_device(self, array: np.ndarray):
        """Convert numpy array to MPS tensor (or CPU tensor as fallback)."""
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("PyTorch not installed")

        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        if self.use_mps:
            return tensor.to(self.device)
        return tensor

    def to_numpy(self, tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if hasattr(tensor, 'device') and tensor.device.type == 'mps':
            tensor = tensor.cpu()
        return tensor.numpy()

    def empty_cache(self):
        """Free MPS memory cache."""
        if self.use_mps:
            torch = _get_torch()
            if torch:
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_gb': mem.total / (1024 ** 3),
                'available_gb': mem.available / (1024 ** 3),
                'used_gb': mem.used / (1024 ** 3),
                'usage_percent': mem.percent,
                'total': mem.total / (1024 ** 3),
                'available': mem.available / (1024 ** 3),
                'used': mem.used / (1024 ** 3),
            }
        except ImportError:
            return {
                'total_gb': self.memory_gb if self.memory_gb else 0,
                'available_gb': 0, 'used_gb': 0, 'usage_percent': 0,
                'total': self.memory_gb if self.memory_gb else 0,
                'available': 0, 'used': 0,
            }

    # ── GPU-Accelerated Feature Extraction ──────────────────────────

    def compute_stft(self, audio: np.ndarray, n_fft: int = 2048,
                     hop_length: int = 512) -> np.ndarray:
        """GPU-accelerated STFT magnitude. Returns numpy array."""
        if not self.use_mps:
            import librosa
            return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

        torch = _get_torch()
        try:
            audio_t = self.to_device(audio)
            if audio_t.ndim > 1:
                audio_t = torch.mean(audio_t, dim=1)
            audio_t = audio_t.contiguous()
            window = torch.hann_window(n_fft, device=self.device)
            with torch.no_grad():
                pad = n_fft // 2
                audio_padded = torch.nn.functional.pad(
                    audio_t, (pad, pad), mode='reflect')
                frames = audio_padded.unfold(0, n_fft, hop_length) * window
                stft_result = torch.fft.rfft(
                    frames, n=n_fft, dim=-1).T
            # torch.stft output is 2x librosa.stft magnitude due to
            # different default window normalization; scale to match.
            stft_result = stft_result * 0.5
            magnitude = torch.abs(stft_result)
            return self.to_numpy(magnitude)
        except Exception:
            import librosa
            return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    def compute_chroma(self, stft_magnitude: np.ndarray,
                       sr: int = 44100) -> np.ndarray:
        """GPU-accelerated chroma from STFT magnitude."""
        if not self.use_mps:
            import librosa
            return librosa.feature.chroma_stft(S=stft_magnitude, sr=sr)

        torch = _get_torch()
        try:
            n_fft = (stft_magnitude.shape[0] - 1) * 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            S = self.to_device(stft_magnitude)
            chroma_fb = np.zeros((12, stft_magnitude.shape[0]), dtype=np.float32)
            for i, freq in enumerate(freqs):
                if freq > 0:
                    pitch_class = int(round(12 * np.log2(freq / 440.0))) % 12
                    chroma_fb[pitch_class, i] += 1.0
            chroma_fb_t = self.to_device(chroma_fb)

            chroma = torch.mm(chroma_fb_t, S)
            result = self.to_numpy(chroma)
            # Normalize
            max_val = result.max(axis=0, keepdims=True) + 1e-8
            return result / max_val

        except Exception:
            import librosa
            return librosa.feature.chroma_stft(S=stft_magnitude, sr=sr)

    def phase_correlation(self, audio1: np.ndarray, audio2: np.ndarray,
                          window_samples: int, sr: int):
        """GPU-accelerated cross-correlation for beat alignment."""
        if not self.use_mps:
            return self._cpu_phase_correlation(audio1, audio2, window_samples, sr)

        torch = _get_torch()
        try:
            # Ensure mono
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)

            seg_len = min(len(audio1), len(audio2), int(sr * 2.0))
            seg1 = self.to_device(audio1[-seg_len:]).contiguous()
            seg2 = self.to_device(audio2[:seg_len]).contiguous()

            # Normalize
            seg1 = seg1 / (torch.max(torch.abs(seg1)) + 1e-8)
            seg2 = seg2 / (torch.max(torch.abs(seg2)) + 1e-8)

            # FFT-based correlation on GPU
            n = seg1.shape[0] + seg2.shape[0] - 1
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*output.*resized.*',
                                        UserWarning)
                fft1 = torch.fft.rfft(seg1, n=n)
                fft2 = torch.fft.rfft(seg2, n=n)
                corr = torch.fft.irfft(fft1 * torch.conj(fft2), n=n)

            corr_np = self.to_numpy(corr)
            center = len(corr_np) // 2
            search_start = max(0, center - window_samples)
            search_end = min(len(corr_np), center + window_samples)
            search_region = corr_np[search_start:search_end]
            peak_idx = np.argmax(search_region)
            optimal_offset = peak_idx + search_start - center

            return corr_np, optimal_offset

        except Exception:
            return self._cpu_phase_correlation(audio1, audio2, window_samples, sr)

    @staticmethod
    def _cpu_phase_correlation(audio1, audio2, window_samples, sr):
        """CPU fallback for cross-correlation."""
        from scipy import signal as sig

        if audio1.ndim > 1:
            audio1 = np.mean(audio1, axis=1)
        if audio2.ndim > 1:
            audio2 = np.mean(audio2, axis=1)

        seg_len = min(len(audio1), len(audio2), int(sr * 2.0))
        seg1 = audio1[-seg_len:]
        seg2 = audio2[:seg_len]

        seg1 = seg1 / (np.max(np.abs(seg1)) + 1e-8)
        seg2 = seg2 / (np.max(np.abs(seg2)) + 1e-8)

        correlation = sig.correlate(seg1, seg2, mode='same', method='fft')
        center = len(correlation) // 2
        search_start = max(0, center - window_samples)
        search_end = min(len(correlation), center + window_samples)
        search_region = correlation[search_start:search_end]
        peak_idx = np.argmax(search_region)
        optimal_offset = peak_idx + search_start - center

        return correlation, optimal_offset

    def __repr__(self):
        return (f"MetalGPU(chip={self.chip_model}, mps={self.use_mps}, "
                f"batch={self.batch_size})")
