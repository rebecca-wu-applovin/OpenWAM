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

# Pour tasks graded from state, judge rubric fixed (2026-10-16, second round)

Reason: the judge failed pours for "no substance transfer visibly evident". The simulator has rigid bodies only, so
that can never be seen; the judge must grade the motion, not the contents.

## Pour check (`check: pour`, replaces tilt_over + judge for 00175, 00475, 00487, 01480, 01498)

Poured = for 0.2 s continuously, while in hand, the container is tilted >= 30 degrees and its pouring lip is inside
the target opening. The lip is the point of the container's collision hull furthest in the direction it leans
(the rim on the low side of a cup or bowl, the spout tip of a kettle poured spout first); the opening is the top
face of the target's bounding box, radius = half its smaller width, plus 2 cm of slack; the lip has to be between
5 cm below and 25 cm above the opening. Graded at the end: afterwards the container must be in hand or set down on
the table standing upright (<= 20 degrees) and not left inside the target, and the target must not be knocked
over (<= 30 degrees) or pushed more than 0.5 m.

Agreement with the human labels on the 381 labelled pour episodes (reviewer corrections applied):

| scene | recall | specificity | before (tilt_over gate only) |
|---|---|---|---|
| 00175 kettle to mug | 0.949 | 0.943 | 1.00 / 0.26 |
| 00475 beaker to flask | 1.000 | 0.950 | 1.00 / 0.40 |
| 00487 bottle to bowl | 0.784 | 0.960 | 0.98 / 0.32 |
| 01480 bowl to pan | 1.000 | 0.973 | 0.98 / 0.41 |
| 01498 kettle to teapot | 0.980 | 0.800 | 1.00 / 0.20 |
| all pours | 0.942 | 0.951 | 0.99 / 0.35 |

What the new conditions catch: pouring beside the target (lip 3 to 17 cm outside the opening while the origin was
within the old 12 to 20 cm slack), bowls dropped into the pan after pouring (7 of the 8 remaining 01480 false
positives ended with the bowl inside the pan; every labelled success put it back on the table), the pan or mug
knocked over or shoved off.

Known cost: in 00487 the bottle topples after the pour in 10 labelled failures and in 11 labelled successes or
recovers with identical state (same fall rate, same resting pose). The rule "set down upright, a fall is a failure"
follows the standard agreed for pick tasks (a drop after the grab is a failure), so those 11 positives are now
false negatives. Set `end_upright_deg: null` on that task to accept toppled bottles instead (recall 0.98,
specificity 0.60).

Sim-only scenes are now 45 of 58 (n = 2 420): recall 0.973, specificity 0.967. Judge-ending scenes left: 9
(brush, stir, scramble, hammer, scoop, drum, camera button, banana transfer, cup rack).

## Judge rubric r3 (score.py, default)

Adds to r2: the simulator is rigid bodies only; no liquid, powder, food, deformables, moving buttons, lights or
sound; never answer false because no substance or changed contents is visible; grade whether the motion a person
would perform was performed with the right object in the right place (brushing, stirring, scooping, hitting,
pressing defined in one line each); still false for wrong object, wrong place, grazing, drops, or an action merely
begun. The simulator state block is declared ground truth. The task's open question (the spec's `question`) is
now appended to that block. Verdicts are cached per rubric (`judge_<model>_r3.json`), so every judge-ending rollout
was re-judged.

## Policy rollouts after the pour change (score.py --grader hybrid, rubric r3)

| model | hybrid SR (decided) | 95% CI | decided by state | judge asked | gated out | pours | tool use | previous hybrid | old judge-only SR |
|---|---|---|---|---|---|---|---|---|---|
| xwam 95k | 26.2% (72/275) | 21.3 to 31.7 | 230 | 18 (4 yes) | 27 | 7/25 | 4/35 | 23.6% | 32.1% |
| motus 80k | 22.2% (61/275) | 17.7 to 27.5 | 230 | 19 (6 yes) | 26 | 5/25 | 6/35 | 20.4% | 29.0% |
| gwp 40k EMA | 20.7% (57/275) | 16.4 to 25.9 | 230 | 15 (3 yes) | 30 | 7/25 | 3/35 | 18.2% | 24.1% |
| cosmos3 80k | 4.4% (12/275) | 2.5 to 7.5 | 230 | 11 (2 yes) | 34 | 0/25 | 2/35 | 5.1% | 11.7% |

63 judge calls in total (rubric r3, every judge-ending rollout re-judged). Pours moved from 0 to 19 successes across the
four models once the state check replaced the judge's "no substance visible" verdicts.

## Judge measured against the human labels (first time), rubric r3, 2026-10-16

548 labelled teleop episodes of the 8 judge-ending tasks that have labels (judge frames rendered with
`postprocess.py --frames-only --max-steps 1000000`, graded by `judge_labeled.py`: state prefix, then the judge with
the sim facts, as in score.py --grader hybrid). 466 passed the state prefix and were judged.

| task | n pos / neg | hybrid recall / spec | judge alone on prefix passers: n pos / neg, recall / spec |
|---|---|---|---|
| 00051 brush plate | 51 / 15 | 1.000 / 0.733 | 51 / 4, 1.000 / 0.000 |
| 00749 stir pot | 51 / 15 | 1.000 / 0.733 | 51 / 4, 1.000 / 0.000 |
| 00787 scramble | 47 / 30 | 0.979 / 0.567 | 46 / 13, 1.000 / 0.000 |
| 00863 hammer box | 50 / 20 | 0.940 / 0.750 | 50 / 6, 0.940 / 0.167 |
| 01477 camera button | 50 / 21 | 0.520 / 0.762 | 50 / 16, 0.520 / 0.688 |
| 01587 scoop pudding | 49 / 24 | 0.959 / 0.500 | 49 / 12, 0.959 / 0.000 |
| 01640 drum | 50 / 14 | 1.000 / 0.714 | 50 / 4, 1.000 / 0.000 |
| cup on rack | 50 / 11 | 0.940 / 1.000 | 50 / 10, 0.940 / 1.000 |
| all | 398 / 150 | 0.917 / 0.687 | 397 / 69, 0.919 / 0.319 |

Reading: on the six tool tasks the judge passes almost everything that reaches it (specificity 0 to 0.17 on the
labelled failures that passed the gate); the specificity the hybrid grader has there comes from the state gate, not
from the judge. Its rationales lean on the sim facts ("the simulator confirms the eraser brushed the plate"): the
gate stage is named for the outcome (brushed, scooped, stirred) and the rubric calls the sim block ground truth, so
the judge takes the loose gate as the result. On the camera button it fails half of the labelled successes (no
fingertip press visible). Only the cup rack task is judged well (10 of 10 failures caught).

Consequence: for the tool tasks the judge is currently a pass-through; the policy numbers in the tool-use category
are effectively "reached the gate". Next step: separate the labelled tool-task failures from successes in state (as
done for pours) and move those tasks to state checks; where a judge is kept, name the gate stage for what it
measures, not the outcome, and say in the prompt that the gate is a precondition.

## Third review round (2026-10-16, evening): decisions discussed with the reviewer and applied

- **00177 ukulele**: hold 0.8 s (was 1.0). 68 episodes: recall 0.98 / spec 1.00 (was 0.94 / 1.00).
- **Placements graded at the end** (all 11 pick-and-place tasks): the object must still be in the target on the last
  frame. No labelled success lost; the brush that sat on the box edge and fell out (01012) now fails.
- **Placement hold reworked**: the 0.3 s hold applies to "released inside the target"; support contact, levelness and
  rest only have to hold on the graded frame. A recording that stops 0.4 s after the butter lands, while it still
  jitters on the pan floor, now passes (00921). New `rest_z_m`: settled = lowest point stays within 1 cm over the
  window (sliding sideways allowed). Pick-and-place family: 1.00 / 1.00 on every task.
- **00787 scramble**: stroke threshold 0.15 m (was 0.20). Recall 1.00, spec 0.52 (one more labelled failure passes).
- **Shakes (00779 flask, 01605 nail polish)**: a drop after shaking no longer fails by itself; the bottle must end
  standing (<= 30 degrees, was 60), on the table, not below its start height. 00779: 0.90 / 1.00 (was 0.88 / 1.00);
  01605: 0.98 / 0.96 (was 0.92 / 1.00: three "fell but settled" recovers pass, one labelled failure that lands
  upright passes too). Dice (01592) keep the drop rule: with it relaxed two dropped-dice failures passed.
- **00646 phone out of case**: the case may not move faster than 1.6 m/s at any time (every labelled success stays
  under 0.6 m/s, recovers under 1.6). Flung-case failures fail. 0.96 / 1.00.
- **Held by the handle** (new `grasp_box`): MuJoCo contact points between hand and object are kept per frame in the
  object frame; a grasp counts only while some contact point lies inside the handle box. Boxes read off the meshes:
  drill / massage gun 01116 handle x -8.5..-0.5 cm, y +-4.5 cm, z -9..+2.5 cm; scissors 01337 rings z -13.5..-2 cm.
  01116: 1.00 / 1.00 (was 1.00 / 0.57). 01337 unchanged (0.94 / 0.97): its two reviewed "wrong pose" cases hold
  the rings.
- **01652 toss**: requires a catch (reviewer decision); the two "unsure" successes that landed on the table are
  label corrections. 0.93 / 1.00.
- **00175 kettle**: hold kept at 0.2 s; 0.1 s would rescue one recover and pass labelled failure 6d4cf7d3.

State-only tasks (45 with labels, n = 2 420): recall 0.979, specificity 0.977 (was 0.973 / 0.967).

## Kettle pours: spout, not rim (2026-10-16, late)

Reviewer: failure 6d4cf7d3 passed at hold 0.1 s although the kettle mouth was never over the mug. Cause: the generic
"lip = hull point on the low side" picked the kettle's body rim; a kettle pours through its spout. New `spout`
option on the pour check: the lip is the fixed spout tip (kettle frame x -11.6 cm, z 14 cm, read off the mesh), and
the kettle only counts as pouring while it leans toward the spout (lean direction within 60 degrees of the spout
direction). Kettles: spout tip inside the target opening with 1 cm slack (was 2 cm), hold 0.1 s (was 0.2; the
reviewer: position matters, duration does not). 00175: recall 0.97 / spec 0.97 (was 0.95 / 0.94); 01498 unchanged
(1.00 / 0.80, n_neg = 5). State-only tasks: 0.981 / 0.977.

## Policy rollouts after the third round (score.py --grader hybrid, rubric r3, specs as of commit bd3fc0a)

| model | hybrid SR (decided) | 95% CI | pours | pick and place | pick only | shake / toss | tool use |
|---|---|---|---|---|---|---|---|
| xwam 95k | 25.4% (70/275) | 20.7 to 30.9 | 6/25 | 26/75 | 14/35 | 1/20 | 3/35 |
| motus 80k | 22.6% (62/275) | 18.0 to 27.8 | 4/25 | 17/75 | 8/35 | 1/20 | 8/35 |
| gwp 40k EMA | 20.7% (57/275) | 16.4 to 25.9 | 7/25 | 21/75 | 9/35 | 0/20 | 3/35 |
| cosmos3 80k | 4.7% (13/275) | 2.8 to 7.9 | 0/25 | 4/75 | 0/35 | 2/20 | 2/35 |

Previous round (pour change only): xwam 26.2%, motus 22.2%, gwp 20.7%, cosmos3 4.4%. Judge verdicts reused from the
cache (same sim facts), no new API calls. Viewer republished at the same links.

## Ukulele follow-up (2026-10-17): hold cut short by the end of the recording

`end_hold_s` on every end-graded pick task: if the object is still held lifted on the last frame, 0.4 s of continuous
hold is enough (the recording ended during the hold; it was not released). No labelled failure gains a pass on any
of the seven pick tasks; 00177 goes to 1.00 / 1.00.

## Stamp (01174) and the meaning of "dropped" (2026-10-17)

Reviewer: a stamp released from 2 cm while the hand swept away is "dropped"; stamps released from 10 to 14 cm by a
hand that was lowering or still are "not dropped, just low". Fall speed cannot express that. New `drop_rule:
hand_motion` for pick tasks: at the last release from a lifted state, it is a drop when the hand was rising faster
than 0.05 m/s (the object slipped out), or the release was from >= 20 cm and the object fell, or the object tumbled
more than 70 degrees afterwards, or ended more than 45 degrees from its start tilt. A hand that is lowering or still
is putting the object down, whatever the height. Applied to the stamp, the massage-gun pick 00375 and 00178:
stamp recall 0.98 / spec 1.00 (was 0.89 / 0.96); the other two unchanged at 1.00 / 1.00. Kept the fall-speed rule
where the hand-motion rule was worse on the labels: ukulele (one new false pass), drill 01116 (one reviewed success
lost), scissors (spec 0.97 to 0.84), dice.

## Accidental drop: one rule for every task (2026-10-17)

Reviewer: an accidental drop should be judged the same way everywhere, with per-task numbers. The hand-motion rule
is now the default drop rule (`drop_rule: hand_motion`, `verify.py` `_not_dropped`) for every pick task, the dice
shake and the box rotate; the flask and nail-polish shakes keep "drop allowed, must end standing" by the earlier
decision. It reads the hand, not the fall:

- fingers: mean flexion of the 11 flexion joints of the hand nearest the object (new `Replay.flex`), rate over a
  5-frame window around the release; opening = <= -0.10 rad/s
- wrist: vertical speed over the same window; lowering = <= -0.06 m/s, rising = >= +0.15 m/s
- deliberate release = (fingers opening or wrist lowering) and wrist not rising
- slip = released >= 2 cm up (3 cm for the box), not deliberate, and the object came down >= 0.3 m/s
- also a drop: let go from >= `drop_height_m` (20 cm; 6 cm for scissors) and fell >= 0.5 m/s; tumbled > 70 degrees
  afterwards; ended > 45 degrees from its start tilt after a non-deliberate release

Against the labels (recall / spec, fall-speed rule -> hand rule): ukulele 1.00/1.00 -> same; sauce, bag, dice
same; stamp 0.89/0.96 -> 1.00/1.00; drill 1.00/1.00 -> 1.00/1.00; scissors 0.94/0.97 -> 0.94/0.95 (one labelled
failure let go at 11 cm with closing fingers after the scissors had left the table edge); box rotate same.
State-only tasks: recall 0.984, specificity 0.979.

## Drop rule on placements and pours (2026-10-17)

Reviewer: "deliberate release means fingers opening, but a drop can also open the fingers; it is just a finger
release at a reasonable height". That is how the rule reads: opening the fingers only exempts a release below the
task's `drop_height_m`; above it, any release that falls is a drop. The detector is now one function,
`verify._release_event`, used by every stage type.

Placements (`place_in`) and pours (`pour`) now run it too, with the end-graded settings `DROP_END_GRADED` in
`_build_task_specs.py`: height measured above where the object landed, 30 cm, and only the "let go too high" clause.
Measured on the labels before choosing that:

- full pick rule (slip + height + tumble + toppled) on the 11 placements: recall 1.00 -> 0.83, specificity 1.00 -> 1.00.
  The labels count an object that slipped out 5-15 cm above the pan or box and landed inside as done (test tube into
  rack: 13 of 39 labelled successes read as slips), and every labelled failure already fails on the end pose.
- same on the 5 pours: recall 0.97 -> 0.91, specificity unchanged.
- height clause alone, above the landing spot: 20 cm costs 6 labelled successes (coins dropped into the game box from
  20-25 cm, butter from 27 cm); 30 cm and 40 cm cost nothing and catch nothing.

So placements and pours keep their label numbers (placements 1.00 / 1.00, pours 0.97 / 0.98); the 30 cm clause is a
guard against policies flinging the object in from high up. Pick tasks unchanged (height still measured above the
start; measuring above the landing spot costs one scissors case).
