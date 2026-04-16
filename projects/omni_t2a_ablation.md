# Omni T2A Ablation Notes

Slack thread: https://luma-ai.slack.com/archives/C0816AKQ4SJ/p1775963690720579

## Gradient Explosion Fix — Noise Schedule + Softcap

Resolved the gradient explosion issue in Omni T2A training through two changes:

### 1. Less aggressive noise schedule (`sigma_shift`)

The original config inherited `sigma_shift=3.0` from T2I, which pushes the noise distribution heavily toward high sigma (mean=0.83, median=0.87). Switching to `sigma_shift=1.0` (no extra shift) gives a broader, more balanced distribution (mean=0.66, median=0.69) — the model sees a healthier mix of noise levels during training instead of being dominated by near-pure-noise samples.

The shift transform is: `sigma' = (S * sigma) / (1 + (S - 1) * sigma)`

Code: [sigma_shift in BagelT2ALoss](https://github.com/lumalabs/lumaverse/blob/dongguo/omni-t2a-v2/projects/kuma/kuma/projects/omni/audio/losses/bagel_t2a.py#L111-L115)

### 2. Softcap on modulation parameters

Without softcap, grad norm starts rebounding after training for a while, eventually diverging. With softcap=1.0 (bounding shift/scale/gate via `tanh(x/cap)*cap`), training stays stable under identical hyperparameters.

- Without softcap: https://wandb.ai/luma-ai/omni-t2a/runs/nt844aee?nw=nwuserdongguo
- With softcap: https://wandb.ai/luma-ai/omni-t2a/runs/fc7t5z49?nw=nwuserdongguo

Code:

- [softcap function](https://github.com/lumalabs/lumaverse/blob/dongguo/omni-t2a-v2/lib/ursa/ursa/math/numeric.py#L71-L77)
- [modulate/apply_gate usage](https://github.com/lumalabs/lumaverse/blob/dongguo/omni-t2a-v2/lib/ursa/ursa/models/utils.py#L146-L199)
- [T2A config helper](https://github.com/lumalabs/lumaverse/blob/dongguo/omni-t2a-v2/projects/kuma/kuma/projects/omni/audio/configs/t2a.py#L526-L541)

## Open Questions

Across several exploding runs, the grad norm typically starts rebounding (from decreasing to increasing) around step 500 or after a few thousand steps depending on how aggressive the setting is. An interesting observation is that in some of these runs, the grad norm *stops increasing and starts decreasing again* once the learning rate warmup finishes and flattens out. Not clear yet whether there's a non-trivial mechanistic link or if it's coincidental.

We did not do extensive ablations, and it's unclear which of these observations are accidental vs. carrying non-trivial learnings applicable to other domains:

1. The grad norm decrease coinciding with learning rate flattening in exploding runs
2. Softcap prevents grad norm explosion in some settings but not all — e.g. grad norm still explodes with aggressive noise schedule even with softcap + grad clip + small learning rate

More systematic experiments would be needed to disentangle these.

## Next Steps

We've resolved the dataloader efficiency bottleneck, so we can now iterate much faster. The current experiments use a 0.6B backbone where a single node can support a reasonably large batch size, making ablations cheap and fast. Planning to kick off a set of ablation studies over the weekend covering: softcap vs no softcap, warmup schedule, grad clip, and noise schedule variations. Goal is to build better intuition for which stabilization knobs actually matter and carry over to larger-scale runs.
