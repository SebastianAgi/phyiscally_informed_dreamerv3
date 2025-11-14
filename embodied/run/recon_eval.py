from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np


def recon_eval(
    make_agent,
    make_replay_eval,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  agent = make_agent()
  replay_eval = make_replay_eval()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  eval_episodes = defaultdict(elements.Agg)
  eval_epstats = elements.Agg()
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  should_log = elements.when.Clock(args.log_every)
  should_report = elements.when.Clock(args.report_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker, mode):
    episode = eval_episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      eval_epstats.add(result)

  fns = [bind(make_env_eval, i) for i in range(args.envs)]
  driver_eval = embodied.Driver(fns, parallel=(not args.debug))
  driver_eval.on_step(replay_eval.add)
  driver_eval.on_step(bind(logfn, mode='eval'))
  driver_eval.on_step(lambda tran, _: policy_fps.step())
  # Increment the global step counter so the outer loop can progress and
  # reporting / logging clocks observe advancing steps.
  driver_eval.on_step(lambda tran, _: step.increment())

  stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))
  carry_eval = agent.init_report(args.batch_size)

  def reportfn(carry, stream):
    agg = elements.Agg()
    for _ in range(args.report_batches):
      batch = next(stream)
      carry, mets = agent.report(carry, batch)
      agg.add(mets)
    return carry, agg.result()

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay_eval = replay_eval
#   if args.from_checkpoint:
#     elements.checkpoint.load(args.from_checkpoint, dict(
#         agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load(args.from_checkpoint, keys=['agent'])

  print('Start training loop')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  driver_eval.reset(agent.init_policy)
  while step < args.steps:

    if should_report(step):
      print('Evaluation')
      driver_eval.reset(agent.init_policy)
      driver_eval(eval_policy, episodes=args.eval_eps)
      logger.add(eval_epstats.result(), prefix='epstats')
      if len(replay_eval):
        carry_eval, mets = reportfn(carry_eval, stream_eval)
        logger.add(mets, prefix='eval')

    driver_eval(eval_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()
