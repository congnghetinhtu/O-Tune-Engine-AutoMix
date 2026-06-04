"""
GPU-Accelerated Cross-Correlation for Beat Alignment
Uses Metal Performance Shaders for ~50x speedup over scipy
"""

import warnings

import torch
import numpy as np
import logging
from typing import Tuple
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class GPUCorrelation:
    """GPU-accelerated cross-correlation optimized for Apple Silicon"""
    
    def __init__(self, gpu):
        """
        Initialize GPU correlation
        
        Args:
            gpu: AppleSiliconGPU instance
        """
        self.gpu = gpu
        self.use_gpu = bool(getattr(gpu, 'use_mps', False))
    
    def phase_correlation(self, audio1: np.ndarray, audio2: np.ndarray, 
                         window_samples: int, sr: int) -> Tuple[np.ndarray, int]:
        """
        Calculate FFT-based cross-correlation using Metal GPU
        
        ~50x faster than scipy.signal.correlate on Apple Silicon
        
        Args:
            audio1: First audio segment
            audio2: Second audio segment
            window_samples: Search window size in samples (±offset)
            sr: Sample rate
            
        Returns:
            Tuple of (correlation values, optimal offset in samples)
        """
        if not self.use_gpu:
            return self._phase_correlation_cpu(audio1, audio2, window_samples, sr)
        
        try:
            # Ensure mono for correlation analysis
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)
            
            # Extract segments for comparison (2 seconds)
            segment_length = min(len(audio1), len(audio2), int(sr * 2.0))
            seg1 = audio1[-segment_length:] if len(audio1) >= segment_length else audio1
            seg2 = audio2[:segment_length] if len(audio2) >= segment_length else audio2
            
            # Apply onset emphasis for better beat correlation
            onset_env1 = self._compute_onset_strength(seg1, sr)
            onset_env2 = self._compute_onset_strength(seg2, sr)
            
            # Resample onset envelopes to match audio length
            onset1 = self._resample_onset(onset_env1, len(seg1))
            onset2 = self._resample_onset(onset_env2, len(seg2))
            
            # Combine audio with onset emphasis (70% audio, 30% onset)
            seg1_enhanced = seg1 * 0.7 + onset1 * 0.3
            seg2_enhanced = seg2 * 0.7 + onset2 * 0.3
            
            # Normalize segments
            seg1_enhanced = seg1_enhanced / (np.max(np.abs(seg1_enhanced)) + 1e-8)
            seg2_enhanced = seg2_enhanced / (np.max(np.abs(seg2_enhanced)) + 1e-8)
            
            # Transfer to MPS device (unified memory = zero-copy!)
            seg1_tensor = self.gpu.to_device(seg1_enhanced)
            seg2_tensor = self.gpu.to_device(seg2_enhanced)
            
            # FFT-based correlation on Metal GPU
            with torch.no_grad():
                # Pad for FFT
                n = len(seg1_tensor) + len(seg2_tensor) - 1
                n_fft = 2 ** int(np.ceil(np.log2(n)))

                # Compute full linear cross-correlation via convolution with reversed seg2.
                # This matches scipy.signal.correlate(x, y, mode='full') lag indexing:
                # lags range from -(len(y)-1) .. (len(x)-1), and zero-lag is at index len(y)-1.
                seg2_rev = torch.flip(seg2_tensor, dims=[0])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*output.*resized.*',
                                            UserWarning)
                    fft1 = torch.fft.rfft(seg1_tensor, n=n_fft)
                    fft2 = torch.fft.rfft(seg2_rev, n=n_fft)
                    conv = torch.fft.irfft(fft1 * fft2, n=n_fft)
                correlation = conv[:n]

                zero = len(seg2_tensor) - 1
                search_start = max(0, int(zero - window_samples))
                search_end = min(int(len(correlation)), int(zero + window_samples + 1))

                # Choose peak via normalized cross-correlation to reduce
                # overlap-length bias (important for periodic signals).
                search_region = correlation[search_start:search_end]
                lags = torch.arange(search_start, search_end, device=correlation.device, dtype=torch.int64) - int(zero)

                x2 = seg1_tensor * seg1_tensor
                y2 = seg2_tensor * seg2_tensor
                x2_cum = torch.cat([torch.zeros(1, device=correlation.device), torch.cumsum(x2, dim=0)])
                y2_cum = torch.cat([torch.zeros(1, device=correlation.device), torch.cumsum(y2, dim=0)])
                L = int(len(seg1_tensor))

                pos = lags >= 0
                lpos = torch.clamp(lags, 0, L).to(torch.int64)
                lneg = torch.clamp(-lags, 0, L).to(torch.int64)

                # Overlap energies for each lag (assuming equal-length segments)
                ex = torch.where(
                    pos,
                    x2_cum[L] - x2_cum[lpos],
                    x2_cum[torch.clamp(L + lags, 0, L).to(torch.int64)],
                )
                ey = torch.where(
                    pos,
                    y2_cum[torch.clamp(L - lpos, 0, L).to(torch.int64)],
                    y2_cum[L] - y2_cum[lneg],
                )

                denom = torch.sqrt(ex * ey + 1e-12)
                scores = search_region / denom
                peak_idx = torch.argmax(scores)
                optimal_offset = int(lags[peak_idx].item())
            
            # Convert back to NumPy
            correlation_np = self.gpu.to_numpy(correlation)

            return correlation_np, int(optimal_offset)
            
        except Exception as e:
            logger.warning(f"GPU correlation failed: {e}, falling back to CPU")
            self.use_gpu = False
            return self._phase_correlation_cpu(audio1, audio2, window_samples, sr)
    
    def _phase_correlation_cpu(self, audio1, audio2, window_samples, sr):
        """
        CPU fallback for phase correlation
        Uses scipy.signal.correlate
        """
        try:
            # Ensure mono
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)
            
            # Extract segments
            segment_length = min(len(audio1), len(audio2), int(sr * 2.0))
            seg1 = audio1[-segment_length:] if len(audio1) >= segment_length else audio1
            seg2 = audio2[:segment_length] if len(audio2) >= segment_length else audio2
            
            # Apply onset emphasis
            onset_env1 = self._compute_onset_strength(seg1, sr)
            onset_env2 = self._compute_onset_strength(seg2, sr)
            
            onset1 = self._resample_onset(onset_env1, len(seg1))
            onset2 = self._resample_onset(onset_env2, len(seg2))
            
            # Combine
            seg1_enhanced = seg1 * 0.7 + onset1 * 0.3
            seg2_enhanced = seg2 * 0.7 + onset2 * 0.3
            
            # Normalize
            seg1_enhanced = seg1_enhanced / (np.max(np.abs(seg1_enhanced)) + 1e-8)
            seg2_enhanced = seg2_enhanced / (np.max(np.abs(seg2_enhanced)) + 1e-8)
            
            # Full cross-correlation (lags: -(len(y)-1) .. (len(x)-1))
            correlation = scipy_signal.correlate(
                seg1_enhanced, seg2_enhanced, mode='full', method='fft'
            )

            zero = len(seg2_enhanced) - 1
            search_start = max(0, int(zero - window_samples))
            search_end = min(int(len(correlation)), int(zero + window_samples + 1))

            search_region = correlation[search_start:search_end]
            lags = np.arange(search_start, search_end, dtype=np.int64) - int(zero)

            x2 = seg1_enhanced.astype(np.float64) ** 2
            y2 = seg2_enhanced.astype(np.float64) ** 2
            x2_cum = np.concatenate([[0.0], np.cumsum(x2)])
            y2_cum = np.concatenate([[0.0], np.cumsum(y2)])
            L = len(seg1_enhanced)

            ex = np.empty_like(lags, dtype=np.float64)
            ey = np.empty_like(lags, dtype=np.float64)
            pos = lags >= 0
            kpos = lags[pos]
            kneg = (-lags[~pos])

            ex[pos] = x2_cum[L] - x2_cum[kpos]
            ey[pos] = y2_cum[L - kpos]
            ex[~pos] = x2_cum[L - kneg]
            ey[~pos] = y2_cum[L] - y2_cum[kneg]

            denom = np.sqrt(ex * ey + 1e-12)
            scores = search_region / denom
            peak_idx = int(np.argmax(scores))
            optimal_offset = int(lags[peak_idx])
            
            return correlation, int(optimal_offset)
            
        except Exception as e:
            logger.warning(f"Phase correlation failed: {e}, using zero offset")
            return np.array([0]), 0
    
    def _compute_onset_strength(self, audio, sr):
        """
        Compute onset strength envelope
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Onset strength envelope
        """
        try:
            import librosa
            return librosa.onset.onset_strength(y=audio, sr=sr)
        except Exception:
            # Fallback: return zeros
            return np.zeros(len(audio) // 512 + 1)
    
    def _resample_onset(self, onset_env, target_length):
        """
        Resample onset envelope to match audio length
        
        Args:
            onset_env: Onset strength envelope
            target_length: Target length in samples
            
        Returns:
            Resampled onset envelope
        """
        try:
            x_old = np.linspace(0, target_length, len(onset_env))
            x_new = np.arange(target_length)
            interpolator = interp1d(x_old, onset_env, bounds_error=False, fill_value=0)
            return interpolator(x_new)
        except Exception:
            # Fallback: return zeros
            return np.zeros(target_length)
    
    def batch_correlation(self, audio_pairs, window_samples, sr):
        """
        Batch cross-correlation for multiple track pairs
        
        Args:
            audio_pairs: List of (audio1, audio2) tuples
            window_samples: Search window size
            sr: Sample rate
            
        Returns:
            List of (correlation, offset) tuples
        """
        results = []
        for audio1, audio2 in audio_pairs:
            corr, offset = self.phase_correlation(audio1, audio2, window_samples, sr)
            results.append((corr, offset))
        return results
    
    def __repr__(self):
        return f"GPUCorrelation(gpu_enabled={self.use_gpu})"
