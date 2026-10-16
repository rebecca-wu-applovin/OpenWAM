# Verifier changes after the human review round (2026-10-16)

Input: 111 review decisions saved from this viewer (`reviews.json`): 74 "human label right, verifier wrong",
35 "verifier right, label wrong", 2 "task is broken". Every "verifier wrong" case was traced to the condition
that failed, and the rule or threshold was changed only where the simulator state actually separates the
reviewed successes from the reviewed failures. Label corrections are applied automatically by `calibrate.py`.

## Agreement with the human labels

Sim-only scenes (the 40 tasks the simulator decides alone), reviewer corrections applied, n = 2 039 episodes.

| family | before (recall / specificity) | after |
|---|---|---|
| all sim-only | 0.956 / 0.893 | **0.975 / 0.969** |
| pick and place | 0.973 / 0.954 | 0.996 / 0.988 |
| open and close | 0.993 / 1.000 | 1.000 / 1.000 |
| caps and lids | 1.000 / 1.000 | 1.000 / 1.000 |
| pick up only | 0.928 / 0.937 | 0.949 / 0.939 |
| flip and rotate | 0.948 / 0.769 | 0.970 / 0.908 |
| shake and toss | 0.813 / 0.688 | 0.867 / 0.987 |

Without the label corrections (raw operator labels): 0.967 / 0.956. Judge-ending scenes are scored here on their
state prefix only (pour 1.00 / 0.42, tool use 0.99 / 0.56): the prefix is meant to be permissive, the judge decides.

Of the 74 "verifier wrong" cases, 57 now agree. The 17 that still disagree:
- 01116 drill x2: "held by the handle", "not stably held". State sees a lifted, held drill either way.
- 01337 scissors: "wrong pose to hold". Same.
- 01246 bowl flip, 01032 box flip: "too brutal". Peak speed and travel of these two are inside the range of
  labelled successes (1.8 m/s, 7 cm travel); three other "too brutal" cases were caught by the new limits.
- 01564 tablet x3: "does not stay in portrait" / "not steady". At the end the tablet is flat, still and at
  74 to 88 degrees from its start, identical to the labelled successes.
- 01174 stamp x3: one "dropped" from 4.6 cm at 0.42 m/s, two "not dropped, height is low" from 10 and 14 cm at
  1.0 m/s. No rule passes the second pair and fails the first.
- 01541 box x2: "flies away" / "uncontrollable rotation". Travel and speed are inside the success range.
- 00779 flask x1 (one reversal only), 00921 butter x2 (placement never still for 0.3 s), 01337 scissors x1
  (dropped from 19 cm at 1.7 m/s, labelled recover).

## Rule changes (verify.py)

1. **Lift measured at the object's lowest point** for pick tasks. A ukulele tipped up on one end is not lifted
   (00177 "part of the object always on the ground"). Drop detection uses the slower of the origin's and the
   lowest point's descent, so a standing stamp falling over is not a drop.
2. **One drop rule for pick, shake and rotate tasks** (`_not_dropped`): pass if the object is still in hand at the
   end, else the last release from a lifted state must not be a drop (>= 8 cm up and falling >= 0.5 m/s, or any
   fall >= 1 m/s) and the object must not end below its start height or off the table. Optional: standing
   upright at the end (flask, nail polish, 60 degrees).
3. **Shake graded at the end**: shaken at some point (3 reversals, was 4) and not dropped afterwards. This removes
   the "bottle falls in the end" passes (00779 x4, 01605 x3, 01592 x1).
4. **Flip and rotate: "not thrown"** limits, each enforced from the moment it is exceeded: peak speed <= 2.1 m/s,
   travel <= 0.55 m (0.40 m for the small box), lowest point never above 0.40 m. Labelled successes stay under
   these; the reviewed "too brutal" cases at 2.6 to 4.2 m/s or 0.5 to 1.0 m of travel fail.
5. **Toss**: tossed = the coin rose >= 4 cm and >= 2 cm away from the wrist within 0.6 s; caught = at the end it
   is in a hand (contact, or within 20 cm of a wrist) and >= 5 cm above the table. The old descent window missed
   quick catches; the old 22 cm catch radius accepted coins lying on the table next to the hand.
6. **Placements**: hold 0.3 s (was 0.5) with dropouts up to 0.25 s bridged; `center_xy_m` option (bread on bread
   centred within 4 cm, the reviewed "not overlapping" cases were 6 to 7 cm off); at-rest tolerance 3 to 4 cm over
   the hold window for butter and egg.
7. **Out of a container** (phone out of case): lifted more than 10 cm above the container counts as out.
8. **Grasp** hold 0.4 s (was 0.5); **pour gates** hold 0.1 s over a 20 cm footprint slack (operators pour from
   beside the target).
9. Every failed stage reports the condition that failed with the measured value (`why` in verifier.json, the red
   box in this viewer, and the sentence given to the judge).

## Task definition changes (_build_task_specs.py)

- 00019 bread on bread: centred within 4 cm.
- 00051 brush, 00787 scramble, 01587 scoop, 00863 hammer: looser gates (5 cm / 20 cm / 2 cm of stroke, hammer
  strike at any speed); the judge decides the outcome.
- 00175, 01498, 00475, 00487, 01480 pours: gate = tilted >= 30 to 50 degrees within 12 to 20 cm of the target.
- 00177 ukulele: hold 1.0 s at the lowest point, drop threshold 10 cm.
- 00807 egg: pan footprint is the dish area where every labelled success ended (x within 7 cm, y -9.5 to 5.5 cm).
- 01116 drill: hold 0.6 s. 01174 / 01304 / 01337 / 00375 / 00178: hold 0.8 s, graded at the end.
- 01541 rotate box: no grasp stage ("rotate does not need grasp"), graded at the end, not thrown.
- 01542 teacup: no reach stage (the cup may already face the robot).
- 01564 tablet: at rest at the end.
- 01032 / 01041 / 01246 flips: not thrown.
- 00779 / 01592 / 01605 shakes: end-state, bottles standing. 01652 toss: end-state.
- Excluded from grading (`skip`): 00381 flower out of box, 00636 teapot lid off (flagged broken in review),
  01000 burger box (broken lid mesh).

## Policy rollouts (hybrid grading, score.py --grader hybrid)

Best checkpoints, 290 rollouts each (58 tasks x 5 seeds); 15 per model belong to the 3 excluded tasks.
Judge = gpt-6-luna, called only when the state prefix passed, with the simulator facts in its prompt.

| model | hybrid SR (decided) | 95% CI | decided by state | judge asked | gated out | old judge-only SR |
|---|---|---|---|---|---|---|
| xwam 95k | 23.6% (65/275) | 19.0 to 29.0 | 205 | 36 (4 yes) | 34 | 32.1% |
| motus 80k | 20.4% (56/275) | 16.0 to 25.5 | 205 | 27 (6 yes) | 43 | 29.0% |
| cosmos3 80k | 5.1% (14/275) | 3.1 to 8.4 | 205 | 11 (4 yes) | 59 | 11.7% |
| gwp 40k EMA | 18.2% (50/275) | 14.1 to 23.2 | 205 | 28 (3 yes) | 42 | 24.1% |

102 judge calls in total. Published viewer (all four models, hybrid verdicts, old judge verdict shown per seed):
https://storage.cloud.google.com/foundational-research/rebecca/sharpa_b64_best/viewer/index_auth.html
The previous judge-only pages are kept as index_judge_only.html and index_auth_judge_only.html in the same folder. The drop against the old judge-only numbers comes from the articulated and grasp-only
tasks the judge used to over-accept (lids moved a few degrees, laptops left closed, objects dropped after the grab)
and from the gate: 136 of the 210 pour and tool-use rollouts never reached the pouring or scrubbing step.
