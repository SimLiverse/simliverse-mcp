# What the palletising cell does outside the geometry it was measured on

Measured 2026-09-03 on an L4 worker, Isaac Sim 6.0.1, via `demo/scenario_sweep.py`.
Two cartons per scenario, one knob changed each. Deterministic: repeated trials
agreed to a tenth of a millimetre.

## Results

| scenario | placed | errors (mm) | s/carton |
|---|---|---|---|
| baseline | 2/2 | 3.0, 5.1 | 58.69 |
| large carton (0.22 m) | 2/2 | 4.1, 0.9 | 66.27 |
| fast belt (0.4 m/s) | 2/2 | 4.2, 3.9 | 58.45 |
| near pallet (0.68 m) | 2/2 | 2.8, 1.6 | 60.70 |
| far pallet (0.95 m) | 2/2 | 2.5, 4.8 | 58.69 |
| 2x3 pattern | 2/2 | 2.1, 2.9 | 58.69 |
| high deck (0.55 m) | 2/2 | 12.0, 5.1 | 76.75 |
| light carton (0.25 kg) | 2/2 | 4.3, **32.9** | 58.02 |
| slow belt (0.1 m/s) | 1/2 | 4.1, **131.9** | 65.30 |
| small carton (0.10 m) | 0/2 | 1754.0 | — |
| heavy carton (3 kg) | 0/2 | 100.0 | — |
| guided belt | 0/2 | 208.2 | — |
| low deck (0.35 m) | 0/2 | 188.7 | — |

Reach was not a limit at either end, and belt speed was not a limit at 2x or
0.5x. Both were predicted to fail and did not, which is the useful direction
to be wrong in.

## The pick has never been centred

This is the finding that matters, and the cell's own accuracy number hid it.

Ablation on the baseline cell, two trials each:

```
guides on,  grip 0.06   0/2   [208.2]
guides on,  grip 0.10   0/2   [208.2]
guides off, grip 0.06   2/2   [3.0, 5.1]
guides off, grip 0.10   2/2   [3.0, 5.1]
```

The arm reaches its commanded pose within 1.6 mm in every one of those runs.
What differs is where the carton is hanging when it arrives: 107 mm off-centre
with side rails fitted, 1.5 mm without.

A carton sealed off-centre **slides into line under the cup while it is being
lifted**. Rails stop it sliding. So the 3-5 mm this cell reports is partly the
carton correcting the approach on its own way up - an accident, not a result,
and not visible from the number.

Two contributing measurements:

- the cup is offset about 18 mm laterally from the `ee_link` axis it is
  commanded through (+62.8 mm along the approach, which is the intended
  standoff; -11.2 / +14.4 mm across it, which is not)
- second cartons are consistently worse than first ones - 4.3 to 32.9 mm,
  4.1 to 131.9 mm - which is the same error compounding as the queue advances

Centring the pick is the single fix most of the red rows above are waiting on.

## Working envelope, stated honestly

- carton **0.15 - 0.22 m**, about **1 kg**
- belt **0.1 - 0.4 m/s**, deck **0.45 - 0.55 m**
- pallet **0.68 - 0.95 m** from the arm base

Outside that the cell does not currently work, and the sweep keeps those rows
red rather than dropping them.

## Side guides: available, off by default

`Conveyor.build(guides=True)` fits rails. A queue accumulating against a hard
stop otherwise squirts cartons sideways - 117 mm off the centre-line at 1 kg,
609 mm at 1.5 kg, on a 400 mm belt - and that is what fails every carton at
1.5 kg and above.

They are off by default because turning them on trades a failure at 1.5 kg for
a failure at every mass, for the reason above. They should go on as soon as the
approach lands centred by itself.

## Two failures that were not what they looked like

**"Cup did not seal after 10 descents."** The carton was not there. `here` was
measured once before the first descent and the loop drove the cup at that point
ten times; a carton that moves gets shoved further by each attempt, aimed
further out still. At 1.0 kg nothing moves and it never showed. At 1.5 kg ten
descents walked the carton off the belt. Now re-read every attempt.

**`near pallet` at 0.60 m: MotionError, wrist at its stop.** Not a reach
problem. A pallet is 1.21 m long and placed by its centre, so 0.60 puts its near
edge at -0.005 and the arm's base inside the pallet. `build` records the overlap
now and the sweep reports it as an impossible cell rather than failing code.
