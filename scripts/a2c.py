"""Train or evaluate GRU-A2C against the unchanged Isaac Lab 2.1 task."""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import traceback


def main():
    started = time.monotonic()
    import torch
    import torch._dynamo  # Resolve Torch typing dependencies before Kit's bundled packages.
    import h5py  # Load matching HDF5 DLLs before GUI extensions.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=16000000)
    parser.add_argument('--max_hours', type=float, default=8.)
    parser.add_argument('--save_every', type=int, default=262144)
    parser.add_argument('--rollouts', type=int, default=512)
    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--gsde', action='store_true')
    parser.add_argument('--noise_period', type=int, default=30)
    parser.add_argument('--trace_every', type=int, default=30)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='runs/a2c/recurrent32')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume_latest', action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (min(args.num_envs, args.total_steps, args.save_every, args.episodes) < 1
            or not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error('counts and max_hours must be positive')
    if args.mode == 'eval' and (not args.checkpoint or args.num_envs != 1):
        parser.error('evaluation requires --checkpoint and --num_envs 1')
    if args.rollouts < 1 or args.sequence_length < 1 or args.rollouts % args.sequence_length:
        parser.error('rollouts must be positive and divisible by sequence_length')
    if args.noise_period < 1 or args.trace_every < 0:
        parser.error('noise_period must be positive and trace_every nonnegative')
    output = Path(args.output)
    if args.resume_latest and (output / 'latest.pt').exists():
        args.checkpoint = str(output / 'latest.pt')
    if args.mode == 'train' and (output / 'latest.pt').exists() and not args.checkpoint:
        parser.error('output already contains a model: use --resume_latest or a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output / 'STOP'
    stream = (output / ('metrics.jsonl' if args.mode == 'train' else 'evaluation.jsonl')).open('a', encoding='utf-8')

    def emit(event, **values):
        line = json.dumps(dict(event=event, timestamp_utc=datetime.now(timezone.utc).isoformat(), **values),
                          allow_nan=False)
        if event != 'trajectory':
            print('[A2C] ' + line, flush=True)
        stream.write(line + '\n')
        stream.flush()

    app = env = agent = None
    previous_handler = None
    interrupted = False
    failed = False
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        args.kit_args = args.kit_args or '--/app/vulkan=false --enable omni.physx.fabric-106.5.3'
        options = {}
        if not args.headless:
            AppLauncher._SIM_APP_CFG_TYPES['create_new_stage'] = [bool]
            options = dict(create_new_stage=False, sync_loads=False)
        launcher = AppLauncher(args, **options)
        app = launcher.app
        if not args.headless:
            import omni.usd
            if omni.usd.get_context().get_stage() is None:
                omni.usd.get_context().new_stage()
        from jetbot_navigation.agents.a2c import A2CConfig, IsaacA2C
        config = A2CConfig(rollouts=args.rollouts, sequence_length=args.sequence_length,
                           gsde=args.gsde, noise_period=args.noise_period)
        from jetbot_navigation.environment import isaac_env_211 as task
        cfg = task.JetbotNavigationIsaacEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
        cfg.lidar_debug_vis = not args.headless
        contract = json.loads(json.dumps(dict(navigation=asdict(cfg.navigation), layout=asdict(cfg.layout),
                              scene_sha256=hashlib.sha256(task.NAVIGATION_USD.read_bytes()).hexdigest())))
        env = task.JetbotNavigationIsaacEnv(cfg, render_mode=None if args.headless else 'human')
        states = env.reset(seed=args.seed)[0]['policy']
        agent = IsaacA2C(config, args.num_envs, args.device, training=args.mode == 'train')
        if args.checkpoint:
            agent.load(args.checkpoint, contract, training=args.mode == 'train')
        (output / 'config.json').write_text(json.dumps(dict(arguments=vars(args), a2c=asdict(config),
                  contract=contract, implementation='skrl==1.4.3',
                  exploration_implementation='stable-baselines3==2.7.0' if args.gsde else 'skrl GaussianMixin'),
                  indent=2), encoding='utf-8')
        emit('startup', mode=args.mode, steps=agent.steps, updates=agent.updates,
             num_envs=args.num_envs, resumed=bool(args.checkpoint), original_task=True,
             gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')

        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
            print('[A2C] Stop requested; saving after the current rollout update.', flush=True)

        previous_handler = signal.signal(signal.SIGINT, stop)
        returns = torch.zeros(args.num_envs, device=args.device)
        lengths = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
        recent_returns, recent_success = deque(maxlen=100), deque(maxlen=100)
        reasons = {str(i): 0 for i in range(1, 5)}
        episodes = vector_steps = 0
        session_steps = 0
        next_save = (agent.steps // args.save_every + 1) * args.save_every
        next_progress = agent.steps + max(args.num_envs * 64, 1000)
        motion = torch.zeros(4, device=args.device)
        motion_frames = 0
        previous_command = None
        closest = float('inf')
        deadline = started + args.max_hours * 3600
        while app.is_running():
            if args.mode == 'train' and agent._rollout % config.rollouts == 0:
                interrupted = interrupted or stop_file.exists()
                if interrupted or time.monotonic() >= deadline or agent.steps >= args.total_steps:
                    break
            elif args.mode == 'eval' and (episodes >= args.episodes or interrupted or time.monotonic() >= deadline):
                break
            with torch.no_grad():
                states = states.clone()
                if args.mode == 'train':
                    actions = agent.act(states, vector_steps, args.total_steps)[0]
                else:
                    actions = agent.evaluate(states)
                next_obs, reward, terminated, truncated, info = env.step(actions.clamp(-1, 1))
                next_states = next_obs['policy']
                done = terminated | truncated
                terminal = info['terminal_observation']
                speed = terminal[:, 19] * torch.where(terminal[:, 19] >= 0,
                                                      cfg.navigation.forward_speed,
                                                      cfg.navigation.reverse_speed)
                command = actions.clamp(-1, 1)
                motion[0] += speed.abs().sum()
                motion[1] += (speed.abs() < 0.01).sum()
                if previous_command is not None:
                    motion[2] += (command[:, 1] * previous_command[:, 1] < 0).sum()
                    motion[3] += (command[:, 0] * previous_command[:, 0] < 0).sum()
                previous_command = command.clone()
                motion_frames += args.num_envs
                returns += reward
                lengths += 1
                distance = info['terminal_observation'][:, 16] * cfg.navigation.goal_distance_scale
                closest = min(closest, float(distance.min()))
                if args.trace_every and vector_steps % args.trace_every == 0:
                    diagnostic = env.diagnostic_tensors()
                    emit('trajectory', mode=args.mode, steps=agent.steps, env_id=0,
                         position=diagnostic['last_position_local_m'][0].tolist(),
                         velocity=diagnostic['last_velocity'][0].tolist(),
                         policy_observation=states[0].tolist(),
                         latent_action=actions[0].tolist(), command=command[0].tolist(),
                         applied_action=diagnostic['last_applied_action'][0].tolist(),
                         goal_distance=float(distance[0]), end_reason=int(info['end_reason'][0]))
                if args.mode == 'train':
                    agent.record_transition(states, actions, reward[:, None], next_states,
                                            terminated[:, None], truncated[:, None], info,
                                            vector_steps, args.total_steps)
                    agent.steps += args.num_envs
                else:
                    agent.reset_evaluation(done)
                for index in done.nonzero().flatten().tolist():
                    reason = int(info['end_reason'][index])
                    episodes += 1
                    reasons[str(reason)] += 1
                    value = float(returns[index])
                    recent_returns.append(value)
                    recent_success.append(reason == 2)
                    emit('episode', mode=args.mode, steps=agent.steps, episode=episodes,
                         env_id=index, return_value=value, length=int(lengths[index]), end_reason=reason)
                returns[done] = 0
                lengths[done] = 0
            if args.mode == 'train':
                agent.post_interaction(vector_steps, args.total_steps)
            vector_steps += 1
            session_steps += args.num_envs
            states = next_states
            if args.mode == 'train' and agent.steps >= next_save and agent._rollout % config.rollouts == 0:
                agent.save(output / 'latest.pt', contract)
                agent.save(output / f'step-{agent.steps}.pt', contract)
                emit('checkpoint', steps=agent.steps, path=str(output / 'latest.pt'))
                next_save = agent.steps + args.save_every
            if agent.steps >= next_progress or (args.mode == 'eval' and vector_steps % 300 == 0):
                elapsed = time.monotonic() - started
                emit('progress', mode=args.mode, steps=agent.steps, session_steps=session_steps,
                     updates=agent.updates, elapsed_seconds=round(elapsed, 1),
                     steps_per_second=round(session_steps / elapsed, 1),
                     remaining_session_seconds=round(max(0, deadline - time.monotonic()), 1),
                     episodes=episodes, recent_success_rate=sum(recent_success) / len(recent_success) if episodes else None,
                     mean_return=sum(recent_returns) / len(recent_returns) if episodes else None,
                     closest_goal_distance=closest, metrics=agent.metrics,
                     mean_absolute_speed_mps=float(motion[0] / motion_frames),
                     stationary_fraction=float(motion[1] / motion_frames),
                     yaw_sign_change_fraction=float(motion[2] / motion_frames),
                     linear_sign_change_fraction=float(motion[3] / motion_frames))
                motion.zero_()
                motion_frames = 0
                next_progress = agent.steps + max(args.num_envs * 64, 1000)
                print(f'[A2C] {agent.steps} transitions | {agent.updates} updates | '
                      f'elapsed {elapsed / 3600:.2f} h | '
                      f'session remaining {max(0, deadline - time.monotonic()) / 3600:.2f} h', flush=True)
        if args.mode == 'train':
            # An externally closed app can interrupt a rollout; its partial batch is not reusable.
            discarded = agent._rollout % config.rollouts * args.num_envs
            agent._rollout -= agent._rollout % config.rollouts
            agent.save(output / 'latest.pt', contract)
            emit('checkpoint', steps=agent.steps, path=str(output / 'latest.pt'))
            if interrupted and stop_file.exists():
                stop_file.unlink()
        else:
            discarded = 0
        reached_target = agent.steps >= args.total_steps if args.mode == 'train' else episodes >= args.episodes
        budget_reached = time.monotonic() >= deadline
        stop_reason = ('target_reached' if reached_target else 'interrupted' if interrupted
                       else 'session_budget' if budget_reached else 'app_closed')
        emit('summary', mode=args.mode, steps=agent.steps, updates=agent.updates,
             session_steps=session_steps, episodes=episodes, end_reasons=reasons,
             recent_success_rate=sum(recent_success) / len(recent_success) if episodes else None,
             mean_return=sum(recent_returns) / len(recent_returns) if episodes else None,
             discarded_partial_rollout_transitions=discarded, elapsed_seconds=round(time.monotonic() - started, 1),
             stop_reason=stop_reason, reached_total_steps=agent.steps >= args.total_steps,
             completed_requested_run=reached_target or (args.mode == 'train' and budget_reached))
    except Exception as error:
        failed = True
        emit('fatal', message=str(error), traceback=traceback.format_exc())
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
        stream.close()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
