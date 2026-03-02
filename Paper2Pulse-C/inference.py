"""
ECG Signal Classifier - Inference Script

Takes digitized ECG signals and predicts diagnostic classes with confidence scores.

Usage:
    # From numpy array
    python inference.py --checkpoint best_model.pt --signal signal.npy
    
    # From WFDB file
    python inference.py --checkpoint best_model.pt --wfdb record_name
    
    # Interactive mode
    python inference.py --checkpoint best_model.pt --interactive
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

# Try to import wfdb for reading WFDB files
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False


# Constants
EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 5000
SAMPLING_RATE = 500
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BIT_NAN_16 = -32768


class ECGClassifier:
    """
    ECG Signal Classifier for inference.
    
    Loads a trained model and provides methods for:
    - Signal validation and preprocessing
    - Single and batch inference
    - Confidence score interpretation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the classifier.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            confidence_threshold: Default threshold for binary predictions
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        self._load_checkpoint()
        
        print(f"ECG Classifier initialized:")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.label_names}")
        print(f"  Thresholds: {dict(zip(self.label_names, [f'{t:.3f}' for t in self.thresholds]))}")
    
    def _load_checkpoint(self):
        """Load model checkpoint and configuration."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get configuration
        self.label_names = checkpoint.get('label_names', [f'Class_{i}' for i in range(5)])
        self.num_classes = len(self.label_names)
        
        # Get thresholds (use optimized if available, else default)
        if 'thresholds' in checkpoint:
            self.thresholds = checkpoint['thresholds']
        elif 'optimal_thresholds' in checkpoint:
            self.thresholds = checkpoint['optimal_thresholds']
        else:
            self.thresholds = np.ones(self.num_classes) * self.confidence_threshold
        
        if isinstance(self.thresholds, torch.Tensor):
            self.thresholds = self.thresholds.cpu().numpy()
        
        # Get model args
        args = checkpoint.get('args', {})
        d_model = args.get('d_model', 256)
        num_layers = args.get('num_layers', 6)
        num_heads = args.get('num_heads', 8)
        downsample_factor = args.get('downsample_factor', 10)
        
        # Import and create model
        from model import SignalClassifier
        
        self.model = SignalClassifier(
            num_classes=self.num_classes,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=EXPECTED_SAMPLES,
            downsample_factor=downsample_factor
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get normalization stats if available
        self.signal_mean = np.array(args.get('signal_mean', [0.0] * 12), dtype=np.float32)
        self.signal_std = np.array(args.get('signal_std', [1.0] * 12), dtype=np.float32)
        
        # Try to load from metadata if checkpoint doesn't have them
        if np.allclose(self.signal_mean, 0) and np.allclose(self.signal_std, 1):
            self._try_load_normalization_stats()
    
    def _try_load_normalization_stats(self):
        """Try to load normalization stats from metadata.json."""
        # Look for metadata.json in common locations
        possible_paths = [
            self.checkpoint_path.parent / 'metadata.json',
            self.checkpoint_path.parent.parent / 'metadata.json',
            Path('data_folder/ptbxl/ground_truth/metadata.json'),
            Path('processed_data/metadata.json'),
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    metadata = json.load(f)
                self.signal_mean = np.array(
                    metadata.get('signal_mean_per_lead', [0.0] * 12), dtype=np.float32
                )
                self.signal_std = np.array(
                    metadata.get('signal_std_per_lead', [1.0] * 12), dtype=np.float32
                )
                print(f"  Loaded normalization stats from: {path}")
                return
        
        print("  Warning: Using default normalization (mean=0, std=1)")
    
    def validate_signal(self, signal: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate input signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            (is_valid, list of warning/error messages)
        """
        messages = []
        is_valid = True
        
        # Check dimensions
        if signal.ndim != 2:
            messages.append(f"ERROR: Expected 2D array, got {signal.ndim}D")
            is_valid = False
            return is_valid, messages
        
        # Check shape
        if signal.shape[1] != EXPECTED_LEADS:
            messages.append(f"ERROR: Expected {EXPECTED_LEADS} leads, got {signal.shape[1]}")
            is_valid = False
        
        if signal.shape[0] != EXPECTED_SAMPLES:
            messages.append(f"WARNING: Expected {EXPECTED_SAMPLES} samples, got {signal.shape[0]}")
            if signal.shape[0] < 100:
                messages.append("ERROR: Signal too short (< 100 samples)")
                is_valid = False
        
        # Check for NaN
        nan_count = np.isnan(signal).sum()
        nan_pct = 100 * nan_count / signal.size
        if nan_pct > 0:
            messages.append(f"INFO: Signal contains {nan_pct:.1f}% NaN values (will be replaced with 0)")
        
        # Check for Inf
        inf_count = np.isinf(signal).sum()
        if inf_count > 0:
            messages.append(f"WARNING: Signal contains {inf_count} Inf values (will be replaced)")
        
        # Check value range (typical ECG is -5 to +5 mV)
        valid_signal = signal[~np.isnan(signal) & ~np.isinf(signal)]
        if len(valid_signal) > 0:
            min_val, max_val = valid_signal.min(), valid_signal.max()
            if abs(min_val) > 50 or abs(max_val) > 50:
                messages.append(f"WARNING: Signal range [{min_val:.2f}, {max_val:.2f}] seems unusual for mV")
        
        return is_valid, messages
    
    def preprocess_signal(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess signal for inference.
        
        Steps:
        1. Ensure correct shape
        2. Replace NaN/Inf with 0
        3. Normalize using training statistics
        4. Pad/truncate to expected length
        
        Args:
            signal: Raw signal [T, 12] or [12, T]
            
        Returns:
            Preprocessed tensor [1, 5000, 12]
        """
        signal = np.array(signal, dtype=np.float32)
        
        # Handle transposed input
        if signal.shape[0] == EXPECTED_LEADS and signal.shape[1] != EXPECTED_LEADS:
            signal = signal.T  # [12, T] -> [T, 12]
        
        # Replace NaN and Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad or truncate to expected length
        if signal.shape[0] < EXPECTED_SAMPLES:
            padding = np.zeros((EXPECTED_SAMPLES - signal.shape[0], signal.shape[1]), dtype=np.float32)
            signal = np.vstack([signal, padding])
        elif signal.shape[0] > EXPECTED_SAMPLES:
            signal = signal[:EXPECTED_SAMPLES]
        
        # Ensure correct number of leads
        if signal.shape[1] < EXPECTED_LEADS:
            padding = np.zeros((signal.shape[0], EXPECTED_LEADS - signal.shape[1]), dtype=np.float32)
            signal = np.hstack([signal, padding])
        elif signal.shape[1] > EXPECTED_LEADS:
            signal = signal[:, :EXPECTED_LEADS]
        
        # Normalize
        signal = (signal - self.signal_mean) / (self.signal_std + 1e-8)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # [1, 5000, 12]
        
        return tensor.to(self.device)
    
    def predict(
        self,
        signal: np.ndarray,
        return_all_scores: bool = False
    ) -> Dict:
        """
        Run inference on a single signal.
        
        Args:
            signal: Input signal [T, 12]
            return_all_scores: If True, return scores for all classes
            
        Returns:
            Dict with:
                - predictions: List of (class_name, confidence) tuples for positive predictions
                - all_scores: Dict of all class scores (if return_all_scores=True)
                - binary_predictions: Binary prediction vector
        """
        # Validate
        is_valid, messages = self.validate_signal(signal)
        for msg in messages:
            print(f"  {msg}")
        
        if not is_valid:
            raise ValueError("Signal validation failed")
        
        # Preprocess
        x = self.preprocess_signal(signal)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply thresholds
        binary_preds = (probs > self.thresholds).astype(int)
        
        # Build results
        predictions = []
        for i, (name, prob, pred) in enumerate(zip(self.label_names, probs, binary_preds)):
            if pred == 1:
                predictions.append((name, float(prob)))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'predictions': predictions,
            'binary_predictions': binary_preds,
            'num_positive': int(binary_preds.sum())
        }
        
        if return_all_scores:
            result['all_scores'] = {name: float(prob) for name, prob in zip(self.label_names, probs)}
        
        return result
    
    def predict_batch(self, signals: List[np.ndarray]) -> List[Dict]:
        """Run inference on multiple signals."""
        return [self.predict(s) for s in signals]
    
    def predict_from_wfdb(self, record_path: str) -> Dict:
        """
        Run inference on a WFDB record.
        
        Args:
            record_path: Path to WFDB record (without extension)
            
        Returns:
            Prediction results dict
        """
        if not HAS_WFDB:
            raise ImportError("wfdb library required. Install with: pip install wfdb")
        
        # Read record
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.astype(np.float32)
        
        # Also read digital signal to detect NaN sentinel
        record_raw = wfdb.rdrecord(record_path, physical=False)
        d_signal = record_raw.d_signal
        
        # Replace sentinel values with NaN (will be handled in preprocess)
        nan_mask = (d_signal == BIT_NAN_16)
        signal[nan_mask] = np.nan
        
        return self.predict(signal, return_all_scores=True)


def load_signal_from_file(file_path: str) -> np.ndarray:
    """
    Load signal from various file formats.
    
    Supported formats:
    - .npy: NumPy array
    - .npz: NumPy compressed (expects 'signal' key)
    - .csv: CSV file (no header, comma-separated) OR with header for lead columns
    - .txt: Text file (whitespace-separated)
    - .json: JSON file with lead arrays
    - directory: Individual lead files (I.csv, II.csv, etc.)
    """
    path = Path(file_path)
    
    # Check if it's a directory with individual lead files
    if path.is_dir():
        from assemble_ecg import load_digitized_leads
        return load_digitized_leads(path)
    
    if path.suffix == '.npy':
        return np.load(file_path)
    
    elif path.suffix == '.npz':
        data = np.load(file_path)
        if 'signal' in data:
            return data['signal']
        else:
            # Return first array
            return data[list(data.keys())[0]]
    
    elif path.suffix == '.json':
        from assemble_ecg import load_digitized_leads
        return load_digitized_leads(path, file_format='json')
    
    elif path.suffix == '.csv':
        # Check if it's a full 5000x12 array or needs assembly
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32, max_rows=1)
        if len(data) == 12:
            # Likely full signal format
            full_data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
            if full_data.shape == (5000, 12):
                return full_data
        # Try assembly
        from assemble_ecg import load_digitized_leads
        return load_digitized_leads(path, file_format='csv')
    
    elif path.suffix == '.txt':
        return np.loadtxt(file_path, dtype=np.float32)
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def format_predictions(result: Dict, verbose: bool = True) -> str:
    """Format prediction results as a readable string."""
    lines = []
    
    if result['num_positive'] == 0:
        lines.append("No conditions detected above threshold.")
    else:
        lines.append(f"Detected {result['num_positive']} condition(s):")
        lines.append("")
        for name, conf in result['predictions']:
            bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
            lines.append(f"  {name:<10} {bar} {conf*100:5.1f}%")
    
    if verbose and 'all_scores' in result:
        lines.append("")
        lines.append("All class scores:")
        for name, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
            bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
            lines.append(f"  {name:<10} {bar} {score*100:5.1f}%")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='ECG Signal Classifier - Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--signal', type=str, default=None,
                        help='Path to signal file (.npy, .npz, .csv, .txt, .json)')
    parser.add_argument('--leads_dir', type=str, default=None,
                        help='Path to directory with individual lead files (I.csv, II.csv, etc.)')
    parser.add_argument('--wfdb', type=str, default=None,
                        help='Path to WFDB record (without extension)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata.json for normalization stats')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Default confidence threshold')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all class scores')
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("=" * 60)
    print("ECG Signal Classifier - Inference")
    print("=" * 60)
    
    classifier = ECGClassifier(
        checkpoint_path=args.checkpoint,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    # If metadata path provided, load normalization stats
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
        classifier.signal_mean = np.array(metadata.get('signal_mean_per_lead', [0.0]*12), dtype=np.float32)
        classifier.signal_std = np.array(metadata.get('signal_std_per_lead', [1.0]*12), dtype=np.float32)
        print(f"  Loaded normalization stats from: {args.metadata}")
    
    print()
    
    # Run inference
    if args.signal:
        print(f"Loading signal from: {args.signal}")
        signal = load_signal_from_file(args.signal)
        print(f"Signal shape: {signal.shape}")
        
        result = classifier.predict(signal, return_all_scores=args.verbose)
        print()
        print(format_predictions(result, verbose=args.verbose))
    
    elif args.leads_dir:
        print(f"Loading digitized leads from: {args.leads_dir}")
        from assemble_ecg import load_digitized_leads
        signal = load_digitized_leads(args.leads_dir)
        print(f"Assembled signal shape: {signal.shape}")
        
        result = classifier.predict(signal, return_all_scores=args.verbose)
        print()
        print(format_predictions(result, verbose=args.verbose))
    
    elif args.wfdb:
        print(f"Loading WFDB record: {args.wfdb}")
        result = classifier.predict_from_wfdb(args.wfdb)
        print()
        print(format_predictions(result, verbose=True))
    
    elif args.interactive:
        print("Interactive mode. Enter file paths to classify (Ctrl+C to exit).")
        print()
        
        while True:
            try:
                file_path = input("Enter signal file path: ").strip()
                if not file_path:
                    continue
                
                if file_path.endswith('.hea') or file_path.endswith('.dat'):
                    # WFDB file
                    record_path = file_path.replace('.hea', '').replace('.dat', '')
                    result = classifier.predict_from_wfdb(record_path)
                else:
                    signal = load_signal_from_file(file_path)
                    result = classifier.predict(signal, return_all_scores=True)
                
                print()
                print(format_predictions(result, verbose=True))
                print()
                
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    else:
        print("No input specified. Use --signal, --leads_dir, --wfdb, or --interactive")
        print()
        parser.print_help()


if __name__ == '__main__':
    main()