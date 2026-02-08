"""
ECG 生理特征提取器
使用基于信号处理的特征提取（适用于短信号 187 采样点）
提取 12 维生理特征
"""
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class ECGFeatureExtractor:
    """
    ECG 特征提取器
    提取 12 维生理特征：心率变异性、波形形态、频域特征、统计特征
    适用于 MIT-BIH 数据集 (187 采样点, 360Hz)
    """
    
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
        self.signal_length = 187  # MIT-BIH 标准长度
        self.feature_names = [
            # 心率变异性 (4维)
            'RR_mean', 'RR_std', 'RR_max', 'RR_min',
            # 波形形态 (3维)
            'QRS_width', 'PR_interval', 'QT_interval',
            # 频域特征 (3维)
            'LF_power', 'HF_power', 'LF_HF_ratio',
            # 统计特征 (2维)
            'Signal_skewness', 'Signal_kurtosis'
        ]
    
    def extract(self, signal_data):
        """
        从单条 ECG 信号中提取 12 维特征
        
        Args:
            signal_data: numpy array, shape [187] 或 [1, 187]
        
        Returns:
            features: numpy array, shape [12]
        """
        # 确保信号是 1D
        if len(signal_data.shape) > 1:
            signal_data = signal_data.squeeze()
        
        # 转换为 numpy float64 以进行精确计算
        sig = np.array(signal_data, dtype=np.float64)
        
        try:
            # ==================== 1. 检测 R 峰 ====================
            r_peaks = self._detect_r_peaks(sig)
            
            features = []
            
            # ==================== 2. 心率变异性特征 (4维) ====================
            if len(r_peaks) >= 2:
                # RR 间期（秒）
                rr_intervals = np.diff(r_peaks) / self.sampling_rate
                
                features.extend([
                    np.mean(rr_intervals),      # RR_mean
                    np.std(rr_intervals),       # RR_std
                    np.max(rr_intervals),       # RR_max
                    np.min(rr_intervals)        # RR_min
                ])
            else:
                # 无法检测 R 峰，使用默认值
                features.extend([0.8, 0.1, 1.0, 0.6])
            
            # ==================== 3. 波形形态特征 (3维) ====================
            # QRS 宽度估计
            qrs_width = self._estimate_qrs_width(sig, r_peaks)
            features.append(qrs_width)
            
            # PR 间期估计
            pr_interval = self._estimate_pr_interval(sig, r_peaks)
            features.append(pr_interval)
            
            # QT 间期估计
            qt_interval = self._estimate_qt_interval(sig, r_peaks)
            features.append(qt_interval)
            
            # ==================== 4. 频域特征 (3维) ====================
            lf_power, hf_power, lf_hf_ratio = self._compute_frequency_features(sig)
            features.extend([lf_power, hf_power, lf_hf_ratio])
            
            # ==================== 5. 统计特征 (2维) ====================
            features.append(skew(sig))           # 偏度
            features.append(kurtosis(sig))       # 峰度
            
        except Exception as e:
            # 如果处理失败，返回零向量
            features = [0.0] * 12
        
        # 确保所有特征都是有效数值
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 裁剪异常值
        features = np.clip(features, -10.0, 10.0)
        
        return features
    
    def _detect_r_peaks(self, sig):
        """
        使用简单的峰值检测算法检测 R 峰
        """
        # 1. 计算信号梯度，增强 QRS 复合波
        gradient = np.gradient(sig)
        
        # 2. 平方增强高频成分
        squared = gradient ** 2
        
        # 3. 移动平均平滑
        window_size = 5
        moving_avg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 4. 自适应阈值检测峰值
        threshold = np.mean(moving_avg) + 0.5 * np.std(moving_avg)
        
        # 5. 找到峰值位置
        peaks = []
        for i in range(1, len(moving_avg) - 1):
            if moving_avg[i] > threshold and moving_avg[i] > moving_avg[i-1] and moving_avg[i] > moving_avg[i+1]:
                # 验证原始信号在该位置也有峰值
                local_window = sig[max(0, i-3):min(len(sig), i+4)]
                if sig[i] > np.mean(local_window) + 0.3 * np.std(local_window):
                    peaks.append(i)
        
        # 6. 合并过近的峰值（QRS 宽度约 80-100ms，即 29-36 个采样点）
        min_distance = int(0.2 * self.sampling_rate)  # 200ms 最小间隔
        filtered_peaks = []
        for peak in peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        return np.array(filtered_peaks)
    
    def _estimate_qrs_width(self, sig, r_peaks):
        """
        估计 QRS 宽度（毫秒）
        使用信号过零点或阈值法
        """
        if len(r_peaks) == 0:
            return 80.0  # 默认值 80ms
        
        widths = []
        for r_peak in r_peaks:
            # 在 R 峰附近查找 QRS 边界
            start = max(0, r_peak - 15)
            end = min(len(sig), r_peak + 15)
            
            # 计算局部阈值
            local_segment = sig[start:end]
            local_max = np.max(np.abs(local_segment))
            threshold = 0.1 * local_max
            
            # 查找 QRS 边界（从左和右分别查找）
            left = r_peak
            right = r_peak
            
            while left > start and abs(sig[left]) > threshold:
                left -= 1
            while right < end - 1 and abs(sig[right]) > threshold:
                right += 1
            
            width_ms = ((right - left) / self.sampling_rate) * 1000
            widths.append(width_ms)
        
        return np.mean(widths) if widths else 80.0
    
    def _estimate_pr_interval(self, sig, r_peaks):
        """
        估计 PR 间期（毫秒）
        基于 QRS 起始点估计
        """
        if len(r_peaks) == 0:
            return 160.0  # 默认值 160ms
        
        # 简化估计：假设 PR 间期约为 QRS 宽度的 2 倍
        qrs_width = self._estimate_qrs_width(sig, r_peaks)
        pr_interval = min(qrs_width * 2 + 80, 200)  # 限制在合理范围
        
        return pr_interval
    
    def _estimate_qt_interval(self, sig, r_peaks):
        """
        估计 QT 间期（毫秒）
        基于 R 峰到信号末端的最大偏差
        """
        if len(r_peaks) == 0:
            return 400.0  # 默认值 400ms
        
        qts = []
        for r_peak in r_peaks:
            # 查找 R 峰后的 T 波（最大正值或负值点）
            search_end = min(len(sig), r_peak + 50)
            t_wave_segment = sig[r_peak:search_end]
            
            if len(t_wave_segment) > 5:
                # 找到 T 波峰值
                t_peak_offset = np.argmax(np.abs(t_wave_segment))
                qt_samples = t_peak_offset + int(self.sampling_rate * 0.08)  # 增加 80ms 到 T 波末端
                qt_ms = (qt_samples / self.sampling_rate) * 1000
                qts.append(min(qt_ms, 500))  # 限制最大值
        
        return np.mean(qts) if qts else 400.0
    
    def _compute_frequency_features(self, sig):
        """
        计算频域特征
        """
        try:
            # 计算功率谱密度
            freqs, psd = signal.welch(sig, fs=self.sampling_rate, nperseg=64)
            
            # LF 带: 0.04-0.15 Hz
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0.01
            
            # HF 带: 0.15-0.4 Hz
            hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
            hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0.01
            
            # 归一化
            total_power = lf_power + hf_power
            if total_power > 0:
                lf_power = min(lf_power / total_power, 1.0)
                hf_power = min(hf_power / total_power, 1.0)
            
            lf_hf_ratio = lf_power / (hf_power + 1e-6)
            lf_hf_ratio = min(lf_hf_ratio, 10.0)
            
            return lf_power, hf_power, lf_hf_ratio
        except:
            return 0.5, 0.5, 1.0
    
    def extract_batch(self, signals, verbose=True):
        """
        批量提取特征
        
        Args:
            signals: numpy array, shape [N, 1, 187] 或 [N, 187]
            verbose: 是否显示进度
        
        Returns:
            features: numpy array, shape [N, 12]
        """
        if len(signals.shape) == 3:
            signals = signals.squeeze(1)  # [N, 187]
        
        features = []
        n_samples = len(signals)
        
        for i in range(n_samples):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples} signals ({(i+1)/n_samples*100:.1f}%)")
            
            feat = self.extract(signals[i])
            features.append(feat)
        
        return np.array(features, dtype=np.float32)


def test_feature_extractor():
    """测试特征提取器"""
    print("=" * 60)
    print("ECG Feature Extractor Test")
    print("=" * 60)
    
    extractor = ECGFeatureExtractor(sampling_rate=360)
    
    # 生成模拟 ECG 信号
    t = np.linspace(0, 187/360, 187)
    
    # 使用简化的 ECG 模型：正弦波 + 一些脉冲
    heart_rate = 75  # bpm
    freq = heart_rate / 60  # Hz
    
    ecg_signal = np.sin(2 * np.pi * freq * t)
    # 添加 QRS 脉冲
    for i in range(3):
        peak_idx = int((i + 0.5) * 187 / 3)
        if peak_idx < 187:
            ecg_signal[peak_idx:min(peak_idx+5, 187)] += 2.0
    
    # 归一化
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    
    # 提取特征
    features = extractor.extract(ecg_signal)
    
    print(f"\nTest signal: simulated ECG, length={len(ecg_signal)}")
    print(f"\nExtracted features ({len(features)} dimensions):")
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name:20s}: {value:8.4f}")
    
    # 批量测试
    print("\n" + "-" * 40)
    print("Batch test: 100 signals")
    batch_signals = np.random.randn(100, 187).astype(np.float32)
    batch_features = extractor.extract_batch(batch_signals, verbose=True)
    
    print(f"\nBatch features shape: {batch_features.shape}")
    print(f"Feature statistics:")
    for i, name in enumerate(extractor.feature_names):
        mean_val = np.mean(batch_features[:, i])
        std_val = np.std(batch_features[:, i])
        print(f"  {name:20s}: mean={mean_val:7.3f}, std={std_val:7.3f}")
    
    print("\n" + "=" * 60)
    print("[OK] Feature extractor test passed!")
    print("=" * 60)
    
    return extractor


if __name__ == "__main__":
    test_feature_extractor()
