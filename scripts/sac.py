"""Train, resume or deterministically evaluate SAC in the real Isaac task."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module, metadata
import json
import hashlib
import math
import shutil
import signal
from pathlib import Path
import time
import traceback


def duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def main():
    import torch
    import torch._dynamo  # Load Torch's typing dependencies before Kit's bundled packages.
    import h5py  # Load matching HDF5 DLLs before Kit's GUI extensions.
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--backend', choices=['lab211', 'lab3'], default='lab211')
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--exploration_period', type=int, default=1,
                        help='Training behavior noise resampling interval in control steps.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--replay_capacity', type=int, default=100000)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--updates_per_transition', type=float, default=None,
                        help='Gradient updates per collected transition; overrides updates_per_step.')
    parser.add_argument('--history_length', type=int, default=1)
    parser.add_argument('--history_stride', type=int, default=1)
    parser.add_argument('--recurrent_hidden_dim', type=int, default=64)
    parser.add_argument('--rnd_scale', type=float, default=0.0)
    parser.add_argument('--rnd_decay_steps', type=int, default=10000000)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--initial_alpha', type=float, default=0.2)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--keep_checkpoints', action='store_true')
    parser.add_argument('--compact_checkpoints', action='store_true',
                        help='Numbered checkpoints omit replay; latest.pt remains fully resumable.')
    parser.add_argument('--resume_latest', action='store_true')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--trace_every', type=int, default=0,
                        help='Record env 0 trajectory every N transitions; 0 disables it.')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode_seconds', type=float, default=120.0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--skip_empty_viewport_wait', action='store_true')
    parser.add_argument('--async_assets', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.backend == 'lab211':
        # Keep PhysX and Fabric on the same local bundle.
        if not args.kit_args:
            args.kit_args = '--/app/vulkan=false --enable omni.physx.fabric-106.5.3'
        if not args.headless:
            args.skip_empty_viewport_wait = args.async_assets = True
    positive = ('total_steps', 'num_envs', 'batch_size', 'replay_capacity', 'updates_per_step', 'exploration_period',
                'hidden_dim', 'save_every', 'log_every', 'episodes', 'history_length',
                'history_stride', 'recurrent_hidden_dim', 'rnd_decay_steps')
    if any(getattr(args, key) < 1 for key in positive) or min(args.warmup_steps, args.trace_every) < 0:
        parser.error('counts must be positive and warmup nonnegative')
    if args.replay_capacity < max(args.batch_size, args.num_envs):
        parser.error('replay must fit the batch and all parallel environments')
    if args.updates_per_transition is not None and (not math.isfinite(args.updates_per_transition)
                                                   or args.updates_per_transition <= 0):
        parser.error('updates_per_transition must be finite and positive')
    if not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
        parser.error('episode_seconds must be finite and positive')
    if args.mode == 'eval' and (not args.checkpoint or args.num_envs != 1):
        parser.error('evaluation requires --checkpoint and --num_envs 1')
    output = Path(args.output or ('runs/sac/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if args.resume_latest:
        if args.mode != 'train' or args.output is None:
            parser.error('resume_latest requires training mode and an explicit output directory')
        if (output / 'latest.pt').is_file():
            args.checkpoint = str(output / 'latest.pt')
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / 'metrics.jsonl').open('a' if args.checkpoint else 'w', encoding='utf-8')

    def emit(event, **values):
        record = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      event=event, **values)
        line = json.dumps(record, allow_nan=False)
        if event != 'trajectory':
            print('[SAC] ' + line, flush=True)
        stream.write(line + '\n')
        stream.flush()

    app, env = None, None
    failed = False
    previous_sigint = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        options = {}
        if args.async_assets:
            options['sync_loads'] = False
        if args.skip_empty_viewport_wait:
            if args.backend != 'lab211':
                raise ValueError('empty viewport workaround is specific to lab211')
            AppLauncher._SIM_APP_CFG_TYPES['create_new_stage'] = [bool]
            options['create_new_stage'] = False
        launcher = AppLauncher(args, **options)
        app = launcher.app
        if args.skip_empty_viewport_wait:
            import omni.usd
            if omni.usd.get_context().get_stage() is None:
                omni.usd.get_context().new_stage()
        from jetbot_navigation.agents.sac import SAC, SACConfig, ReplayBuffer, replay_successor, BehaviorNoise
        from jetbot_navigation.agents.sac_features import ObservationHistory, UpdateSchedule
        module = import_module('jetbot_navigation.environment.isaac_env_211' if args.backend == 'lab211'
                               else 'jetbot_navigation.environment.isaac_env')
        cfg = module.JetbotNavigationIsaacEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
        cfg.episode_length_s = cfg.navigation.episode_seconds = args.episode_seconds
        headless = args.headless if args.backend == 'lab211' else not launcher.has_window
        cfg.lidar_debug_vis = not headless
        env = module.JetbotNavigationIsaacEnv(cfg, render_mode=None if headless else 'human')
        agent_config = SACConfig(hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
                                 gamma=args.gamma, tau=args.tau, initial_alpha=args.initial_alpha,
                                 history_length=args.history_length, history_stride=args.history_stride,
                                 recurrent_hidden_dim=args.recurrent_hidden_dim, rnd_scale=args.rnd_scale,
                                 rnd_decay_steps=args.rnd_decay_steps)
        agent = SAC(agent_config, args.device)
        replay = ReplayBuffer(args.replay_capacity, agent_config) if args.mode == 'train' else None
        contract = dict(navigation=asdict(cfg.navigation), layout=asdict(cfg.layout),
                        scene_sha256=hashlib.sha256(module.NAVIGATION_USD.read_bytes()).hexdigest())
        # Canonical JSON representation avoids tuple/list changes after serialization.
        contract = json.loads(json.dumps(contract))
        versions = dict(torch=str(torch.__version__), isaacsim=metadata.version('isaacsim'),
                        isaaclab=metadata.version('isaaclab'), backend=args.backend)
        run_config = dict(arguments=vars(args), sac=asdict(agent_config), contract=contract,
                          versions=versions, replay_device='cpu' if replay is not None else None)
        config_path = output / 'config.json'
        if config_path.exists():
            config_path = output / ('config_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')
        steps = 0
        update_ratio = (args.updates_per_transition if args.updates_per_transition is not None
                        else args.updates_per_step / args.num_envs)
        schedule = UpdateSchedule(update_ratio)
        observation, _ = env.reset(seed=args.seed)
        if args.checkpoint:
            saved = agent.load(args.checkpoint, replay if args.mode == 'train' else None)
            if saved['contract'] != contract:
                raise ValueError('checkpoint task contract differs from this environment')
            steps = saved['steps'] if args.mode == 'train' else 0
            if args.mode == 'train':
                if 'update_ratio' in saved and saved['update_ratio'] != update_ratio:
                    raise ValueError('update ratio differs from checkpoint')
                schedule.credit = saved.get('update_credit', 0.)
        history = ObservationHistory(observation['policy'].detach().clone(), args.history_length, args.history_stride)
        obs = history.current()
        closest_goal_distance = observation['policy'][:, 16].min().item() * cfg.navigation.goal_distance_scale
        if args.checkpoint and args.mode == 'train':
            closest_goal_distance = saved.get('closest_goal_distance_m', closest_goal_distance)
        returns = torch.zeros(args.num_envs, device=args.device)
        lengths = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        behavior_noise = (BehaviorNoise(args.num_envs, 2, args.exploration_period, args.device)
                          if args.mode == 'train' and args.exploration_period > 1 else None)
        completed, rewards, reason_counts = 0, [], {str(key): 0 for key in range(1, 5)}
        episode_reasons = []
        next_log, next_save = steps + args.log_every, steps + args.save_every
        initial_steps = steps
        started = time.monotonic()
        metrics = {}
        saved_steps = None
        stop_requested = False
        emit('startup', mode=args.mode, steps=steps, versions=versions,
             gpu=torch.cuda.get_device_name() if 'cuda' in args.device else None,
             resumed=bool(args.checkpoint), checkpoint=args.checkpoint, output=str(output),
             num_envs=args.num_envs, update_ratio=update_ratio,
             replay_bytes=sum(value.numel() * value.element_size() for value in replay.data.values())
             if replay is not None else 0)

        def save():
            nonlocal saved_steps
            agent.save(output / 'latest.pt', replay, steps=steps, contract=contract,
                       versions=versions, seed=args.seed, update_ratio=update_ratio, update_credit=schedule.credit,
                       closest_goal_distance_m=closest_goal_distance)
            saved_steps = steps
            emit('checkpoint', steps=steps, updates=agent.updates, path=str(output / 'latest.pt'))
            if args.keep_checkpoints:
                snapshot = output / f'step_{steps}.pt'
                if args.compact_checkpoints:
                    agent.save(snapshot, steps=steps, contract=contract, versions=versions,
                               seed=args.seed, update_ratio=update_ratio, update_credit=schedule.credit,
                               closest_goal_distance_m=closest_goal_distance)
                else:
                    temporary = snapshot.with_suffix('.pt.tmp')
                    shutil.copyfile(output / 'latest.pt', temporary)
                    temporary.replace(snapshot)

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        previous_sigint = signal.signal(signal.SIGINT, request_stop)

        while app.is_running() and not stop_requested:
            if args.mode == 'train' and steps >= args.total_steps:
                break
            if args.mode == 'eval' and completed >= args.episodes:
                break
            if args.mode == 'train' and steps < args.warmup_steps:
                actions = (behavior_noise.sample(lengths, uniform=True).clone() if behavior_noise is not None
                           else torch.rand(args.num_envs, 2, device=args.device) * 2 - 1)
            else:
                noise = behavior_noise.sample(lengths) if behavior_noise is not None else None
                actions = agent.act(obs, deterministic=args.mode == 'eval', noise=noise)
            observation, reward, terminated, truncated, info = env.step(actions)
            next_obs = observation['policy'].detach().clone()
            reward, terminated, truncated = reward.clone(), terminated.clone(), truncated.clone()
            if not torch.isfinite(next_obs).all() or not torch.isfinite(reward).all():
                raise FloatingPointError('non-finite environment output')
            raw_successor = replay_successor(next_obs, terminated, truncated, info)
            closest_goal_distance = min(closest_goal_distance,
                                        raw_successor[:, 16].min().item() * cfg.navigation.goal_distance_scale)
            successor, next_history = history.advance(raw_successor, next_obs, terminated | truncated)
            if args.mode == 'train':
                replay.add(obs, actions, reward, successor, terminated)
                if agent.rnd is not None:
                    metrics['rnd_loss'] = agent.rnd.observe(raw_successor)
            steps += args.num_envs
            returns += reward
            lengths += 1
            ended = (terminated | truncated).nonzero(as_tuple=False).flatten().tolist()
            if args.trace_every and (steps == args.num_envs or steps % args.trace_every == 0 or 0 in ended):
                diag = env.diagnostic_tensors()
                position = diag['last_position_local_m'][0].cpu().tolist()
                yaw = diag['last_yaw_rad'][0].item()
                dx, dy = cfg.layout.goal[0] - position[0], cfg.layout.goal[1] - position[1]
                bearing = math.atan2(dy, dx) - yaw
                emit('trajectory', steps=steps, env_id=0, episode_step=lengths[0].item(),
                     position_m=position, yaw_rad=yaw, goal_position_m=list(cfg.layout.goal),
                     goal_distance_m=math.hypot(dx, dy),
                     goal_bearing_rad=math.atan2(math.sin(bearing), math.cos(bearing)),
                     command_action=actions[0].cpu().tolist(),
                     applied_action=diag['last_applied_action'][0].cpu().tolist(),
                     velocity=diag['last_velocity'][0].cpu().tolist(),
                     lidar_min_m=diag['last_lidar_sector_m'][0].min().item(),
                     lidar_sectors_m=diag['last_lidar_sector_m'][0].cpu().tolist(),
                     contact_force_N=diag['last_obstacle_force_N'][0].cpu().tolist(),
                     end_reason=int(info['end_reason'][0].item()),
                     reward_components={key: value[0].item() for key, value in info['reward_components'].items()})
            for index in ended:
                reason = int(info['end_reason'][index].item())
                value = returns[index].item()
                completed += 1
                rewards.append(value)
                episode_reasons.append(reason)
                reason_counts[str(reason)] += 1
                emit('episode', steps=steps, env_id=index, episode=completed,
                     return_value=value, length=int(lengths[index].item()), end_reason=reason)
                returns[index], lengths[index] = 0, 0
            obs = next_history
            if args.mode == 'train' and steps >= args.warmup_steps and replay.size >= args.batch_size:
                for _ in range(schedule.advance(args.num_envs)):
                    metrics.update(agent.update(replay.sample(args.batch_size, agent.device)))
            if steps >= next_log:
                elapsed = max(time.monotonic() - started, 1e-6)
                speed = (steps - initial_steps) / elapsed
                remaining = max(args.total_steps - steps, 0) / speed if speed > 0 and args.mode == 'train' else None
                recent_count = min(completed, 100)
                recent_reasons = episode_reasons[-100:]
                emit('progress', steps=steps, updates=agent.updates,
                     replay_size=replay.size if replay is not None else 0,
                     episodes=completed, mean_return=sum(rewards[-100:]) / len(rewards[-100:]) if rewards else None,
                     recent_success_rate=recent_reasons.count(2) / recent_count if recent_count else None,
                     steps_per_second=speed, elapsed_seconds=elapsed, remaining_seconds=remaining,
                     progress_percent=100 * min(steps / args.total_steps, 1) if args.mode == 'train' else None,
                     num_envs=args.num_envs, update_ratio=update_ratio,
                     closest_goal_distance_m=closest_goal_distance,
                     torch_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1048576
                     if 'cuda' in args.device else 0,
                     **metrics)
                if args.mode == 'train':
                    eta = duration(remaining) if remaining is not None else 'unknown'
                    print(f'[SAC] {steps}/{args.total_steps} ({100 * min(steps / args.total_steps, 1):.1f}%)'
                          f' | elapsed {duration(elapsed)} | remaining ~{eta}', flush=True)
                next_log += args.log_every
            if args.mode == 'train' and steps >= next_save:
                save()
                next_save += args.save_every
        if args.mode == 'train' and saved_steps != steps:
            save()
        if stop_requested:
            emit('interrupted', steps=steps, message='Stopped at a step boundary.')
        emit('summary', mode=args.mode, steps=steps, updates=agent.updates, episodes=completed,
             mean_return=sum(rewards) / len(rewards) if rewards else None,
             success_rate=reason_counts['2'] / completed if completed else None,
             end_reasons=reason_counts, completed_requested_run=(steps >= args.total_steps if args.mode == 'train'
                                                               else completed >= args.episodes))
    except KeyboardInterrupt:
        emit('interrupted', message='Run stopped; resume from the last completed checkpoint.')
    except Exception:
        failed = True
        emit('fatal', traceback=traceback.format_exc())
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        stream.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
