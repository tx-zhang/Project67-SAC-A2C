"""Train in blocks, evaluating each checkpoint before the next block."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def completed_summary(events, mode):
    if any(row['event'] == 'fatal' for row in events):
        raise RuntimeError(f'{mode} failed; inspect metrics.jsonl')
    summaries = [row for row in events if row['event'] == 'summary' and row['mode'] == mode]
    if not summaries or not summaries[-1]['completed_requested_run']:
        raise RuntimeError(f'{mode} did not complete; the last checkpoint remains available')
    result = summaries[-1]
    if mode == 'eval' and (not result['episodes'] or result['mean_return'] is None
                           or not math.isfinite(result['mean_return'])):
        raise RuntimeError('evaluation has no valid completed episodes')
    return result


def assess(history, patience, return_drop):
    """Stop only after repeated degradation from the best evaluated policy."""
    best = None
    worse = 0
    for row in history:
        score = (row['success_rate'], row['mean_return'])
        if best is None or score > (best['success_rate'], best['mean_return']):
            best, worse = row, 0
        elif (row['success_rate'] < best['success_rate']
              or (row['success_rate'] == best['success_rate']
                  and row['mean_return'] < best['mean_return'] - return_drop)):
            worse += 1
        else:
            worse = 0
    return best, worse >= patience


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/sac/gamma-0999')
    parser.add_argument('--total_steps', type=int, default=500000)
    parser.add_argument('--interval', type=int, default=50000)
    parser.add_argument('--eval_episodes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--return_drop', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--exploration_period', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--replay_capacity', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--updates_per_transition', type=float, default=None)
    parser.add_argument('--history_length', type=int, default=1)
    parser.add_argument('--history_stride', type=int, default=1)
    parser.add_argument('--recurrent_hidden_dim', type=int, default=64)
    parser.add_argument('--rnd_scale', type=float, default=0.)
    parser.add_argument('--rnd_decay_steps', type=int, default=10000000)
    parser.add_argument('--min_stop_steps', type=int, default=0)
    parser.add_argument('--compact_checkpoints', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.total_steps, args.interval, args.eval_episodes, args.patience, args.exploration_period,
           args.num_envs, args.batch_size, args.replay_capacity, args.history_length,
           args.history_stride, args.recurrent_hidden_dim, args.rnd_decay_steps, args.hidden_dim) < 1:
        parser.error('step counts, episodes and patience must be positive')
    if min(args.warmup_steps, args.min_stop_steps) < 0 or args.replay_capacity < max(args.batch_size, args.num_envs):
        parser.error('invalid warmup, stop threshold or replay capacity')
    if args.updates_per_transition is not None and (not math.isfinite(args.updates_per_transition)
                                                   or args.updates_per_transition <= 0):
        parser.error('updates_per_transition must be finite and positive')
    if not math.isfinite(args.rnd_scale) or args.rnd_scale < 0:
        parser.error('rnd_scale must be finite and nonnegative')
    if not 0 <= args.gamma < 1 or not math.isfinite(args.return_drop) or args.return_drop <= 0:
        parser.error('gamma must be in [0, 1) and return_drop must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['output'] = str(output)
    config_path = output / 'experiment.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        defaults = dict(exploration_period=1, num_envs=1, warmup_steps=2000, replay_capacity=100000,
                        batch_size=256, hidden_dim=256, updates_per_transition=None, history_length=1,
                        history_stride=1, recurrent_hidden_dim=64, rnd_scale=0.,
                        rnd_decay_steps=10000000, min_stop_steps=0, compact_checkpoints=False)
        for name, value in defaults.items():
            previous.setdefault(name, value)
        if previous != config:
            raise ValueError('experiment settings changed; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    train = output / 'train'
    history_path = output / 'evaluations.jsonl'
    history = read_events(history_path)
    events = read_events(train / 'metrics.jsonl')
    checkpoints = [row['steps'] for row in events if row['event'] == 'checkpoint']
    steps = checkpoints[-1] if checkpoints else 0
    if (train / 'latest.pt').exists() != bool(checkpoints):
        raise ValueError('checkpoint and training log disagree; inspect the output directory')
    common = ['--headless', '--device', args.device,
              '--gamma', str(args.gamma), '--seed', str(args.seed),
              '--exploration_period', str(args.exploration_period)]
    for name in ('warmup_steps', 'replay_capacity', 'batch_size', 'hidden_dim', 'history_length',
                 'history_stride', 'recurrent_hidden_dim', 'rnd_scale', 'rnd_decay_steps'):
        common += ['--' + name, str(getattr(args, name))]
    if args.updates_per_transition is not None:
        common += ['--updates_per_transition', str(args.updates_per_transition)]
    if args.compact_checkpoints:
        common += ['--compact_checkpoints']

    def run(arguments, folder, mode):
        path = folder / 'metrics.jsonl'
        before = len(read_events(path))
        subprocess.run([sys.executable, '-u', '-m', 'scripts.sac', *arguments], cwd=root, check=True)
        return completed_summary(read_events(path)[before:], mode)

    while True:
        best, stop = assess(history, args.patience, args.return_drop)
        if stop and steps >= args.min_stop_steps:
            print('[EXPERIMENT] Stopped after repeated evaluation degradation. Best model: '
                  + str(output / 'best.pt'), flush=True)
            break
        evaluated = bool(history) and history[-1]['checkpoint_steps'] == steps
        if steps and not evaluated:
            folder = output / f'eval_{steps}'
            result = run([*common, '--num_envs', '1', '--mode', 'eval', '--checkpoint', str(train / 'latest.pt'),
                          '--episodes', str(args.eval_episodes), '--trace_every', '30',
                          '--log_every', '1000', '--output', str(folder)], folder, 'eval')
            row = dict(event='evaluation', checkpoint_steps=steps,
                       success_rate=result['success_rate'], mean_return=result['mean_return'],
                       episodes=result['episodes'], end_reasons=result['end_reasons'])
            history.append(row)
            new_best, _ = assess(history, args.patience, args.return_drop)
            if new_best is row:
                temporary = output / 'best.pt.tmp'
                shutil.copyfile(train / 'latest.pt', temporary)
                temporary.replace(output / 'best.pt')
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            print('[EXPERIMENT] ' + json.dumps(row), flush=True)
            continue
        if steps >= args.total_steps:
            print('[EXPERIMENT] Training and checkpoint evaluations completed.', flush=True)
            break
        target = min((steps // args.interval + 1) * args.interval, args.total_steps)
        result = run([*common, '--num_envs', str(args.num_envs), '--total_steps', str(target), '--save_every', str(args.interval),
                      '--keep_checkpoints', '--resume_latest', '--output', str(train)], train, 'train')
        steps = result['steps']


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[EXPERIMENT] Interrupted. Rerun the same task to resume saved progress.', flush=True)
        raise SystemExit(130)
