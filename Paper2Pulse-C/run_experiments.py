"""
Experiment Orchestrator for ECG Augmentation Experiments

Generates all experiment configs (baseline, HP search, ablation) and
runs them in parallel across multiple GPUs using a file-based job queue.

Usage:
    # Generate configs only
    python run_experiments.py --data_dir /path/to/data --generate_only

    # Run everything
    python run_experiments.py --data_dir /path/to/data --num_gpus 2

    # Run specific stage
    python run_experiments.py --data_dir /path/to/data --stage hp_search --num_gpus 2

    # Resume after interruption (skips completed experiments)
    python run_experiments.py --data_dir /path/to/data --num_gpus 2 --resume
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Default training config (speed-optimized for tuning)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TRAINING = {
    'epochs': 30,
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 2,
    'patience': 7,
    'grad_clip': 1.0,
    'num_workers': 4,
}

DEFAULT_MODEL = {
    'd_model': 256,
    'num_layers': 6,
    'num_heads': 8,
    'downsample_factor': 10,
}

DEFAULT_CV = {
    'n_folds': 10,
    'seed': 42,
}

DEFAULT_DATA = {
    'subset_fraction': 1.0,
    'subset_seed': 42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Config generation: Baseline
# ──────────────────────────────────────────────────────────────────────────────

def no_aug():
    """Empty augmentation config (all disabled)."""
    return {
        'amplitude': {'enabled': False},
        'noise': {'enabled': False},
        'lead_mask': {'enabled': False},
        'time_mask': {'enabled': False},
        'time_shift': {'enabled': False},
    }


def generate_baseline(base_dir, data_dir):
    """Generate baseline experiment config (no augmentation)."""
    config = {
        'experiment_id': 'baseline',
        'stage': 'baseline',
        'technique': 'none',
        'description': 'Baseline - no augmentation',
        'data': {**DEFAULT_DATA, 'data_dir': data_dir},
        'augmentation': no_aug(),
        'training': DEFAULT_TRAINING,
        'model': DEFAULT_MODEL,
        'cv': DEFAULT_CV,
        'output_dir': str(base_dir / 'results' / 'baseline'),
    }
    return [config]


# ──────────────────────────────────────────────────────────────────────────────
# Config generation: HP Search
# ──────────────────────────────────────────────────────────────────────────────

def generate_hp_search(base_dir, data_dir):
    """Generate all hyperparameter search configs."""
    configs = []

    # ── Amplitude Scaling ──
    amp_configs = [
        {'scale_min': 0.9, 'scale_max': 1.1},
        {'scale_min': 0.8, 'scale_max': 1.2},
        {'scale_min': 0.85, 'scale_max': 1.15},
        {'scale_min': 0.9, 'scale_max': 1.2},
        {'scale_min': 0.8, 'scale_max': 1.1},
    ]
    for i, hp in enumerate(amp_configs):
        aug = no_aug()
        aug['amplitude'] = {'enabled': True, 'p': 0.5, **hp}
        configs.append({
            'experiment_id': f'hp_amplitude_{i}',
            'stage': 'hp_search',
            'technique': 'amplitude',
            'description': f'Amplitude scaling: [{hp["scale_min"]}, {hp["scale_max"]}]',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'hp_search' / f'amplitude_{i}'),
        })

    # ── Gaussian Noise ──
    noise_configs = [
        {'std': 0.01},
        {'std': 0.02},
        {'std': 0.03},
        {'std': 0.05},
        {'std': 0.10},
    ]
    for i, hp in enumerate(noise_configs):
        aug = no_aug()
        aug['noise'] = {'enabled': True, 'p': 0.5, **hp}
        configs.append({
            'experiment_id': f'hp_noise_{i}',
            'stage': 'hp_search',
            'technique': 'noise',
            'description': f'Gaussian noise: std={hp["std"]}',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'hp_search' / f'noise_{i}'),
        })

    # ── Random Lead Masking ──
    rlm_configs = [
        {'mask_prob': 0.1, 'min_leads_kept': 6},
        {'mask_prob': 0.2, 'min_leads_kept': 6},
        {'mask_prob': 0.3, 'min_leads_kept': 6},
        {'mask_prob': 0.4, 'min_leads_kept': 4},
        {'mask_prob': 0.5, 'min_leads_kept': 4},
    ]
    for i, hp in enumerate(rlm_configs):
        aug = no_aug()
        aug['lead_mask'] = {'enabled': True, 'p': 0.5, **hp}
        configs.append({
            'experiment_id': f'hp_leadmask_{i}',
            'stage': 'hp_search',
            'technique': 'lead_mask',
            'description': f'Lead masking: prob={hp["mask_prob"]}, min_kept={hp["min_leads_kept"]}',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'hp_search' / f'leadmask_{i}'),
        })

    # ── Time Masking ──
    tmask_configs = [
        {'max_length': 50, 'num_masks': 1},
        {'max_length': 100, 'num_masks': 1},
        {'max_length': 200, 'num_masks': 1},
        {'max_length': 100, 'num_masks': 2},
        {'max_length': 250, 'num_masks': 1},
    ]
    for i, hp in enumerate(tmask_configs):
        aug = no_aug()
        aug['time_mask'] = {'enabled': True, 'p': 0.5, **hp}
        configs.append({
            'experiment_id': f'hp_timemask_{i}',
            'stage': 'hp_search',
            'technique': 'time_mask',
            'description': f'Time masking: max_len={hp["max_length"]}, n={hp["num_masks"]}',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'hp_search' / f'timemask_{i}'),
        })

    # ── Time Shifting ──
    tshift_configs = [
        {'max_shift': 25},
        {'max_shift': 50},
        {'max_shift': 100},
        {'max_shift': 150},
        {'max_shift': 200},
    ]
    for i, hp in enumerate(tshift_configs):
        aug = no_aug()
        aug['time_shift'] = {'enabled': True, 'p': 0.5, **hp}
        configs.append({
            'experiment_id': f'hp_timeshift_{i}',
            'stage': 'hp_search',
            'technique': 'time_shift',
            'description': f'Time shifting: max_shift={hp["max_shift"]}',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'hp_search' / f'timeshift_{i}'),
        })

    return configs


# ──────────────────────────────────────────────────────────────────────────────
# Config generation: Ablation
# ──────────────────────────────────────────────────────────────────────────────

def generate_ablation(base_dir, data_dir, best_hps: Dict[str, Dict]):
    """
    Generate additive and subtractive ablation configs.
    
    best_hps: Dict mapping technique name to best HP config from HP search.
        e.g. {'amplitude': {'scale_min': 0.9, 'scale_max': 1.1},
               'noise': {'std': 0.02}, ...}
    """
    configs = []

    # Ordered list of techniques for additive ablation
    techniques = ['amplitude', 'noise', 'lead_mask', 'time_mask', 'time_shift']
    technique_keys = {
        'amplitude': lambda hp: {'enabled': True, 'p': 0.5, **hp},
        'noise': lambda hp: {'enabled': True, 'p': 0.5, **hp},
        'lead_mask': lambda hp: {'enabled': True, 'p': 0.5, **hp},
        'time_mask': lambda hp: {'enabled': True, 'p': 0.5, **hp},
        'time_shift': lambda hp: {'enabled': True, 'p': 0.5, **hp},
    }

    # ── Additive ablation: build up from baseline ──
    # Step 0 is baseline (already have it), steps 1-5 add one technique each
    for step in range(1, len(techniques) + 1):
        aug = no_aug()
        enabled_names = []
        for t_idx in range(step):
            t_name = techniques[t_idx]
            if t_name in best_hps:
                aug[t_name] = technique_keys[t_name](best_hps[t_name])
                enabled_names.append(t_name)

        configs.append({
            'experiment_id': f'abl_add_{step}',
            'stage': 'ablation_additive',
            'technique': '+'.join(enabled_names),
            'description': f'Additive step {step}: +{techniques[step-1]} (enabled: {", ".join(enabled_names)})',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'ablation' / f'additive_{step}'),
        })

    # ── Subtractive ablation: remove one from full ──
    # Full config (all enabled)
    full_aug = no_aug()
    for t_name in techniques:
        if t_name in best_hps:
            full_aug[t_name] = technique_keys[t_name](best_hps[t_name])

    # Full experiment (sub_0)
    configs.append({
        'experiment_id': 'abl_sub_full',
        'stage': 'ablation_subtractive',
        'technique': 'all',
        'description': 'Subtractive baseline: all techniques enabled',
        'data': {**DEFAULT_DATA, 'data_dir': data_dir},
        'augmentation': full_aug,
        'training': DEFAULT_TRAINING,
        'model': DEFAULT_MODEL,
        'cv': DEFAULT_CV,
        'output_dir': str(base_dir / 'results' / 'ablation' / 'subtractive_full'),
    })

    # Remove one at a time
    for t_name in techniques:
        if t_name not in best_hps:
            continue
        aug = json.loads(json.dumps(full_aug))  # deep copy
        aug[t_name] = {'enabled': False}

        configs.append({
            'experiment_id': f'abl_sub_no_{t_name}',
            'stage': 'ablation_subtractive',
            'technique': f'-{t_name}',
            'description': f'Subtractive: remove {t_name}',
            'data': {**DEFAULT_DATA, 'data_dir': data_dir},
            'augmentation': aug,
            'training': DEFAULT_TRAINING,
            'model': DEFAULT_MODEL,
            'cv': DEFAULT_CV,
            'output_dir': str(base_dir / 'results' / 'ablation' / f'subtractive_no_{t_name}'),
        })

    return configs


# ──────────────────────────────────────────────────────────────────────────────
# Best HP selection (reads HP search results)
# ──────────────────────────────────────────────────────────────────────────────

def select_best_hps(base_dir) -> Dict[str, Dict]:
    """
    Read HP search results and select best config per technique.
    Returns dict mapping technique name to best hyperparameters.
    """
    hp_dir = base_dir / 'results' / 'hp_search'
    if not hp_dir.exists():
        print("WARNING: No HP search results found, using defaults")
        return {
            'amplitude': {'scale_min': 0.9, 'scale_max': 1.1},
            'noise': {'std': 0.02},
            'lead_mask': {'mask_prob': 0.2, 'min_leads_kept': 6},
            'time_mask': {'max_length': 100, 'num_masks': 1},
            'time_shift': {'max_shift': 50},
        }

    technique_map = {
        'amplitude': 'amplitude',
        'noise': 'noise',
        'leadmask': 'lead_mask',
        'timemask': 'time_mask',
        'timeshift': 'time_shift',
    }

    best_hps = {}
    best_scores = {}

    for sub_dir in sorted(hp_dir.iterdir()):
        summary_path = sub_dir / 'summary.json'
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        # Determine technique from directory name
        dir_name = sub_dir.name  # e.g. "amplitude_0", "noise_3"
        for prefix, tech_name in technique_map.items():
            if dir_name.startswith(prefix):
                auc_mean = summary.get('auc_mean', 0)
                if tech_name not in best_scores or auc_mean > best_scores[tech_name]:
                    best_scores[tech_name] = auc_mean
                    # Extract the HP values from the augmentation config
                    aug_cfg = summary.get('augmentation', {})
                    hp_cfg = aug_cfg.get(tech_name, {})
                    # Remove non-HP keys
                    hp_only = {k: v for k, v in hp_cfg.items() if k not in ('enabled', 'p')}
                    best_hps[tech_name] = hp_only
                break

    return best_hps


# ──────────────────────────────────────────────────────────────────────────────
# Config saving
# ──────────────────────────────────────────────────────────────────────────────

def save_configs(configs: List[Dict], config_dir: Path) -> List[str]:
    """Save configs to JSON files, return list of paths."""
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cfg in configs:
        path = config_dir / f'{cfg["experiment_id"]}.json'
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        paths.append(str(path))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Job execution
# ──────────────────────────────────────────────────────────────────────────────

def is_experiment_complete(config: Dict) -> bool:
    """Check if experiment has already completed."""
    done_file = Path(config['output_dir']) / 'DONE'
    return done_file.exists()


def run_single_job(config_path: str, gpu_id: int, script_dir: str) -> Dict:
    """Run a single experiment via subprocess, return result info."""
    runner_script = os.path.join(script_dir, 'experiment_runner.py')
    cmd = [
        sys.executable, runner_script,
        '--config', config_path,
        '--gpu', '0',  # Always 0 inside subprocess (CUDA_VISIBLE_DEVICES remaps)
    ]

    # CRITICAL: Set CUDA_VISIBLE_DEVICES so subprocess sees only its assigned GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=None, env=env  
        )
        elapsed = time.time() - start_time

        config_name = Path(config_path).stem
        success = result.returncode == 0

        if not success:
            # Log error to experiment's output dir and fallback logs dir
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                error_dir = Path(cfg['output_dir'])
            except Exception:
                error_dir = Path(config_path).parent.parent / 'logs'
            error_dir.mkdir(parents=True, exist_ok=True)
            with open(error_dir / 'error.log', 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")

        return {
            'config': config_path,
            'config_name': config_name,
            'gpu': gpu_id,
            'success': success,
            'time': elapsed,
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            'config': config_path,
            'config_name': Path(config_path).stem,
            'gpu': gpu_id,
            'success': False,
            'time': time.time() - start_time,
            'returncode': -1,
            'error': 'TIMEOUT',
        }


def run_job_queue(config_paths: List[str], num_gpus: int, script_dir: str,
                  log_dir: Path, resume: bool = False):
    """
    Run all experiments using a round-robin GPU assignment with process pool.
    Each GPU runs one experiment at a time.
    """
    logger = logging.getLogger('orchestrator')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        fh = logging.FileHandler(log_dir / 'orchestrator.log', mode='a')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Filter out already-completed experiments if resuming
    if resume:
        pending = []
        for cp in config_paths:
            with open(cp) as f:
                cfg = json.load(f)
            if is_experiment_complete(cfg):
                logger.info(f"SKIP (done): {cfg['experiment_id']}")
            else:
                pending.append(cp)
        config_paths = pending

    total = len(config_paths)
    if total == 0:
        logger.info("No experiments to run.")
        return []

    logger.info(f"Running {total} experiments on {num_gpus} GPU(s)")
    logger.info(f"Estimated time: ~{total * 5 / num_gpus:.0f} minutes")

    results = []
    completed = 0
    start_time = time.time()

    # Use ProcessPoolExecutor with max_workers = num_gpus
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_info = {}
        for i, cp in enumerate(config_paths):
            gpu_id = i % num_gpus
            future = executor.submit(run_single_job, cp, gpu_id, script_dir)
            future_to_info[future] = {'config': cp, 'gpu': gpu_id, 'index': i}

        for future in as_completed(future_to_info):
            info = future_to_info[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)

                elapsed_total = time.time() - start_time
                avg_per_job = elapsed_total / completed
                remaining = (total - completed) * avg_per_job / max(num_gpus, 1)

                status = "✓" if result['success'] else "✗"
                logger.info(
                    f"[{completed}/{total}] {status} {result['config_name']} "
                    f"GPU{result['gpu']} {result['time']:.0f}s "
                    f"(ETA: {remaining/60:.0f}m)"
                )
            except Exception as e:
                logger.error(f"[{completed}/{total}] ERROR: {info['config']} - {e}")
                results.append({
                    'config': info['config'],
                    'config_name': Path(info['config']).stem,
                    'gpu': info['gpu'],
                    'success': False,
                    'time': 0,
                    'error': str(e),
                })

    total_time = time.time() - start_time
    successes = sum(1 for r in results if r.get('success', False))
    logger.info(f"\nCompleted: {successes}/{total} succeeded in {total_time/60:.1f} minutes")

    # Save run log
    with open(log_dir / 'run_log.json', 'w') as f:
        json.dump({
            'total': total,
            'successes': successes,
            'failures': total - successes,
            'total_time_s': round(total_time, 2),
            'results': results,
        }, f, indent=2)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ECG Augmentation Experiment Orchestrator')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed PTB-XL data directory')
    parser.add_argument('--base_dir', type=str, default='./ecg_experiments',
                        help='Base directory for all experiment outputs')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'baseline', 'hp_search', 'ablation'],
                        help='Which stage to run')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate configs, do not run experiments')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed experiments')
    parser.add_argument('--best_hps_file', type=str, default=None,
                        help='Path to JSON with best HPs (skip auto-detection)')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    config_dir = base_dir / 'configs'
    log_dir = base_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir

    print(f"\n{'='*70}")
    print(f" ECG Augmentation Experiment Orchestrator")
    print(f"{'='*70}")
    print(f"  Data: {data_dir}")
    print(f"  Base: {base_dir}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Stage: {args.stage}")
    print(f"  Resume: {args.resume}")
    print(f"{'='*70}\n")

    all_config_paths = []

    # ── Stage 1: Baseline ──
    if args.stage in ('all', 'baseline'):
        print("Generating baseline config...")
        baseline_cfgs = generate_baseline(base_dir, data_dir)
        paths = save_configs(baseline_cfgs, config_dir / 'baseline')
        all_config_paths.extend(paths)
        print(f"  → {len(paths)} experiment(s)")

    # ── Stage 2: HP Search ──
    if args.stage in ('all', 'hp_search'):
        print("Generating HP search configs...")
        hp_cfgs = generate_hp_search(base_dir, data_dir)
        paths = save_configs(hp_cfgs, config_dir / 'hp_search')
        all_config_paths.extend(paths)
        print(f"  → {len(paths)} experiment(s)")

    # ── Stage 3: Ablation ──
    if args.stage in ('all', 'ablation'):
        # Get best HPs (from file or from HP search results)
        if args.best_hps_file:
            with open(args.best_hps_file) as f:
                best_hps = json.load(f)
            print(f"Loaded best HPs from: {args.best_hps_file}")
        else:
            best_hps = select_best_hps(base_dir)
            # Save for reference
            best_hps_path = base_dir / 'best_hps.json'
            with open(best_hps_path, 'w') as f:
                json.dump(best_hps, f, indent=2)
            print(f"Selected best HPs (saved to {best_hps_path}):")

        for tech, hp in best_hps.items():
            print(f"  {tech}: {hp}")

        print("Generating ablation configs...")
        abl_cfgs = generate_ablation(base_dir, data_dir, best_hps)
        paths = save_configs(abl_cfgs, config_dir / 'ablation')
        all_config_paths.extend(paths)
        print(f"  → {len(paths)} experiment(s)")

    print(f"\nTotal experiments: {len(all_config_paths)}")

    if args.generate_only:
        print("\n--generate_only flag set. Configs saved. Exiting.")
        print(f"Configs: {config_dir}")
        return

    # ── Run experiments ──
    if args.stage == 'all':
        # Run in stages: baseline first, then HP search, then ablation
        print("\n── Stage 1: Baseline ──")
        baseline_paths = [p for p in all_config_paths if 'baseline' in p]
        if baseline_paths:
            run_job_queue(baseline_paths, 1, script_dir, log_dir, args.resume)

        print("\n── Stage 2: HP Search ──")
        hp_paths = [p for p in all_config_paths if 'hp_search' in p]
        if hp_paths:
            run_job_queue(hp_paths, args.num_gpus, script_dir, log_dir, args.resume)

        # Re-select best HPs after HP search completes
        print("\n── Selecting best HPs from search results ──")
        best_hps = select_best_hps(base_dir)
        best_hps_path = base_dir / 'best_hps.json'
        with open(best_hps_path, 'w') as f:
            json.dump(best_hps, f, indent=2)
        for tech, hp in best_hps.items():
            print(f"  {tech}: {hp}")

        # Regenerate ablation configs with actual best HPs
        print("\n── Regenerating ablation configs with best HPs ──")
        abl_cfgs = generate_ablation(base_dir, data_dir, best_hps)
        abl_paths = save_configs(abl_cfgs, config_dir / 'ablation')
        print(f"  → {len(abl_paths)} ablation experiment(s)")

        print("\n── Stage 3: Ablation ──")
        if abl_paths:
            run_job_queue(abl_paths, args.num_gpus, script_dir, log_dir, args.resume)
    else:
        # Run single stage
        run_job_queue(all_config_paths, args.num_gpus, script_dir, log_dir, args.resume)

    print(f"\n{'='*70}")
    print(f"All experiments complete!")
    print(f"Results: {base_dir / 'results'}")
    print(f"Logs: {log_dir}")
    print(f"Run: python aggregate_results.py --base_dir {base_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()