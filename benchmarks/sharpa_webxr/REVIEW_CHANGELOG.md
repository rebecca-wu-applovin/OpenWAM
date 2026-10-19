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

## Scissors: held properly = fingers through the rings (2026-10-17)

Reviewer: two labelled failures that passed were "wrong pose to hold scissor" (7b7acaed, d6000e1d). The handle box
did not catch them: both hold the rings, but pinched from the outside by the middle and ring fingers, blades pointing
straight away from the wrist. Every labelled success holds them with the thumb through one ring and a finger through
the other (29 of 31 successes, the other two thumb only).

New `finger_holes` option on `hold_lifted` (`verify.py` `Replay.fingers_in_holes`, `c_hold_lifted`): the finger
segment origins (PP/MP/DP/fingertip of all five fingers, both hands, new `Replay.fingers`) are mapped into the object
frame; a finger is "through" a hole when a segment lies inside an ellipsoid at the hole. Held properly = thumb in one
hole, or fingers in two holes, while touching. Scissors holes read off the mesh: centres y +-4.4 cm, z -7.9 cm, radii
2.5 / 2.0 / 3.4 cm; the labels give the same result from 2.2 to 3.0 cm.

Measured on the labels (recall / spec): 0.936 / 0.947 -> 0.936 / 1.000. Fixed: 7b7acaed, d6000e1d. Nothing lost.
Still wrong: 7850423f (drop at 19 cm, reviewer sided with the label) and two labelled successes.

Note on method: labelled recordings are verified in full (what the labeller watched); policy rollouts are cut at
the task limit. The placement/pour and first scissors verify runs of this round, launched from the PR worktree,
failed on an import and left old verifier files in place; the quoted label numbers were unaffected. Re-run from the
main tree on the full recordings; all 17 tasks re-verified. State-only tasks: recall 0.986, specificity 0.982.

## Scissors: strict grip, thumb in one ring and index in the other (2026-10-18)

Reviewer: the correct way to hold scissors is the thumb through one ring and the index finger through the other.
`finger_holes` gains a `rule` option (`verify.py` `c_hold_lifted`): `thumb_or_two` (previous default), `thumb_and_finger`
(thumb in one hole, any other finger of the same hand in another) and `thumb_and_index` (same, index only). 01337 now
uses `thumb_and_index`.

Measured on the labels (recall / spec): 0.936 / 1.000 -> 0.894 / 1.000. Newly failing labelled successes, by the
reviewer's choice: recover_dd4f756f (index and middle through the rings, no thumb) and success_4f76159a (thumb through
one ring; the other ring holds the ring finger for 0.4 s, the index sits outside it). State-only tasks: recall 0.984,
specificity 0.982. Policy hybrid SR unchanged (xwam 25.4%, motus 22.6%, gwp 20.4%, cosmos3 4.7%).

Also: when a stage sets both `grasp_box` and `finger_holes`, the failure explanation now names both checks.

## Phone out of case (00646): rotated footprint, no minimum distance (2026-10-18)

Reviewer comments: success_86ba7c00 "moved >= 0.08 m from its start it is strange, it should be just out of box";
success_cc5f5691 "why is this wrong".

- cc5f5691: the phone ends on the table beside the case, but "inside" was tested against the case's world-axis box.
  The case had been turned about 45 deg, so that box covered the table next to it. New `oriented` option on
  `Replay.in_target` (`verify.py`): the footprint is the target's own box, on its two most horizontal axes. Used by
  00646 `move_out`. Tried on all 23 tasks with a target: identical except 00051 and 00787 (one positive lost, one or
  two failures caught each); left at world-axis there for now.
- 86ba7c00: the phone was slid out 7.8 cm (threshold 8 cm). `move_out` takes `min_disp_m: 0` (out of the case alone
  decides), and the grasp needs 3 cm of carry instead of 5 (a phone slid out of the case is carried only a few cm).
  Removing the grasp stage instead lets five phones knocked out of the case pass.

Measured on 00646 labels: recall 0.960 -> 1.000; recover_14ac3297 (phone ends on the table beside the case, reviewed
"verifier right" when it failed) now passes, so it counts against specificity until re-reviewed (0.958).
Policy hybrid SR unchanged.

## Oriented footprint is the default (2026-10-18)

Reviewer: make the rotated footprint the default everywhere. `Replay.in_target` now tests the target's own box on its
two most horizontal axes by default (`oriented=False` keeps the world-axis box). Re-verified all 23 tasks with a
target (1533 labelled recordings in full, 480 policy rollouts at the task cut).

Changes on the labels: 00646 as before (recall 1.000); 00051 loses success_03ac6849 (eraser path over the plate
0.00 m in the rotated box) and catches one more failure; 00787 loses success_ccf05e19 (spatula path in the pan
0.12 m, needs 0.15) and catches two more failures. State-only tasks: recall 0.986, specificity 0.981.
Policy hybrid SR unchanged (xwam 25.4%, motus 22.6%, gwp 20.4%, cosmos3 4.7%).

## Review round 2026-10-18: drops at the end, flung objects, pour collisions (2026-10-18)

New rules (`verify.py`):
- **Not flung** (`_not_flung`, stage key `max_free_speed_m_s`): after the hand lets go for the last time, the object
  never moves faster than the limit. A put-down or a short slip stays slow; a throw, a toss, a box that flies off or a
  tablet that falls flat is fast. Only the last release counts (an object that got away and was picked up again is a
  recover). Labelled positives stay below ~1.1 m/s; the commented failures reach 1.5-2.4 m/s. Limits: 1.2 m/s for
  rotate / flip tasks (01032, 01246, 01541, 01542, 01564), 1.8 m/s for the game board flip (01041, positives toss it
  up to 1.8), 1.0 m/s for the shakes (00779, 01605).
- **Pour without collision** (`c_pour`, `max_hit_speed_m_s`, default 0.7 m/s in `pour()`): the poured object must
  not touch the target while moving faster than 0.7 m/s. Labelled successes touch at <= 0.45 m/s; 01498
  failure_c137b850 hits the teapot at 0.84 m/s. No labelled pour success lost on any of the five pour tasks.
- Flip tasks (01032, 01041, 01246) no longer limit how far the object travels from its start (reviewer: "this should
  not be the condition of upturn"); "not flung" replaces it.
- Optional `toward_target_m` on pours without a spout (pour toward the target: the low lip nearer the opening centre
  than the body). Not enabled: on 01480 it catches failure_358e3d99 only in a 1 cm window of the margin and loses
  recover_305611be there.

Spec changes: 00082 laptop closing speed limit 3.3 -> 4.8 rad/s (reviewer: "speed is not that bad"; costs
failure_7e48c8c2, which closes at 3.6 rad/s and fails for a reason the state does not show); shakes 00779 / 01605
count 2 cm shake strokes (was 3 cm).

Per task (recall / spec): 00082 0.983/1.000 -> 1.000/0.955; 01032 0.975/1.0 -> 1.0/1.0; 01246 1.0/0.8 -> 1.0/1.0;
01541 spec 0.5 -> 1.0; 01564 spec 0.875 -> 0.958; 01605 0.980/0.917 -> 1.0/1.0; 00779 recall 0.92 -> 0.96;
01498 spec 0.8 -> 1.0. State-only tasks: recall 0.988, specificity 0.990. Policy hybrid SR unchanged.

Not fixed: 01564 failure_be78a36c (released 4 cm up, slides 4 cm at 0.8 m/s: not distinguishable from a put-down);
00175 recover_3db6087a (spout tip never within 1 cm of the mug rim while tilted toward the spout: best 3.6 cm out);
00779 recover_186a553e (flask ends lying at 62 deg) and success_b4391e4e; 01480 success_d4aaa0fe has no pan in the
sim (pan 266 m away: scene glitch, excluded as "both wrong").

## 01480: pour toward the pan (2026-10-18)

Reviewer: turn on the "pour toward the target" check for 01480 (`toward_target_m: 0.03`): while pouring, the low rim
must be nearer the pan centre than the bowl's body (+3 cm), unless the bowl is above the middle half of the pan.
failure_358e3d99 (bowl tipped sideways over the pan edge) now fails; recover_305611be is lost.
01480: recall 1.000 -> 0.980, specificity 0.972 -> 1.000. State-only tasks: recall 0.988, specificity 0.994.

## Scissors hold time, shakes ending in the hand, readable explanations (2026-10-18)

- 01337 success_ce440a8c "it is held enough long": a proper scissors grip now needs 0.5 s lifted (was 0.8 s); the
  episode holds it 0.50 s. Nothing lost. success_4f76159a "why it detects held 0 second": the index finger is in a ring
  for only 0.7 s, never while lifted, so the strict thumb-and-index grip is held lifted 0 s. The explanation now lists
  touching / on the handle / proper grip (with thumb and index times) / all together, and names the rule in force
  (it still described the old "thumb in one, or fingers in two" rule).
- 00779 / 01605 "the final bottle is in the hand": a shake ends either in the hand (touching for the whole last 0.5 s,
  at any height: new `held_end_s` and `held_end_lift_m: -0.05`) or standing upright on the table. Fixes
  recover_186a553e (flask held tilted, resting on the table). A single touch on the last frame is not "in hand"
  (failure_07c04993 lies on its side with one fingertip on it). success_b4391e4e stays a failure: the hand lost contact
  0.7 s before the end and the flask tips to 48-55 deg against the open palm.
- Pour explanation rewritten as named parts with times: tilted, low rim over the opening, tipped toward the target
  (01480), in hand, all together for hold_s. 01480 recover_305611be: every part holds, but together only for 0.10 s
  in a row (needs 0.2 s).
State-only tasks: recall 0.997, specificity 0.994 (with the reviewer's latest decisions). Policy hybrid SR unchanged.

## Pours: 0.1 s is enough (2026-10-18)

Reviewer on 01480 recover_305611be: "0.1s is enough". `pour()` now needs the pour parts together for 0.1 s (was
0.2 s; the kettle pours already used 0.1 s). 01480: recall 0.980 -> 1.000, specificity 1.000 -> 0.972: failure_617fef78
now passes (it pours for 0.15 s; the operator also picked up the pan and moved it 0.25 m, within the 0.5 m
"not pushed away" limit). 00475 / 00487 unchanged. State-only tasks: recall 0.998, specificity 0.992.

## Pours: rim inside the opening, correct pan centre (2026-10-18)

Reviewer on 01480 failure_617fef78: "how can this be judged success, it is not even poured". The bowl tips beside the
pan while slipping out of the hand and falls next to it; for 0.15 s its low rim was 1 cm outside the pan rim, which the
2 cm rim slack accepted. Two fixes:
- `pour()` rim slack 2 cm -> 0: the low rim must be inside the opening (kettle pours keep their own 1 cm).
- 01480 pan opening: the pan's box includes the handle, so its centre sat 7 cm toward the handle. The opening is now
  the pan body (centre x 0.07 m in the pan frame, radius 0.158 m, rim top 0.034 m).
01480: 1.000 / 1.000; 00475 and 00487 unchanged. State-only tasks: recall 0.998, specificity 0.994.

## 00051 brushing a held-up plate (2026-10-18)

Reviewer on 00051 success_03ac6849: "it is brushed...". Since the oriented footprint became the default, brushing was
measured inside the plate's footprint on its two most horizontal axes; this plate is held up, tilted and upside down
while the other hand brushes it, so the eraser was never "inside" (0.00 m). New `inside_box_m` option on `sweep`: the
eraser must be on the plate's own box grown by 2.5 cm on every side, whatever the plate's pose. The brushing stage is
now graded at the end and adds the "not flung" rule (1.2 m/s), matching the reviewer's note on success_df3b7731
("brush drops in the end"). Sweep stages can also take the drop rule now.
00051 state prefix (judge decides after it): recall 0.98 -> 1.000; failure_2031b10e and failure_e88912cc (eraser
flung away at 2.1 / 2.4 m/s) now fail before the judge. State-only tasks unchanged (0.998 / 0.994).

## 01564 tablet: not toppled after release (2026-10-18)

Reviewer on failure_be78a36c: "it rotates but it drops in the end". The tablet is stood up on its edge (8 cm), the hand
lets go and it tips over flat on its own, slowly (0.34 m/s), so "not flung" missed it. New `max_free_rot_deg` clause
in `_not_flung`: after the last release the object may turn at most this much on its own. 01564 uses 40 deg (labelled
successes turn <= 37 deg after release, be78a36c 45 deg). Not used on flips, where letting the object turn over is the
task. 01564 all 01564 disagreements now resolved: recall 1.000, specificity 1.000. State-only tasks: 0.998 / 0.995.

## 00787 scramble: path 12 cm, spatula not flung (2026-10-18)

Reviewer on success_ccf05e19: "it is scrambled?". Its spatula path in the pan is 0.12 m; the stage asked 0.15 m. Now
0.12 m (no labelled failure newly passes the state prefix), and the stage adds "not flung" at 1.0 m/s (labelled
successes <= 0.87 m/s after release), which stops four labelled failures before the judge (4f5c7ea4, 5a0525fe,
748fe4ce, c46dc186: spatula thrown down at 1.0-2.3 m/s). 00787 state prefix: recall 1.000, specificity 0.60 -> 0.71.
Eight labelled failures still pass the prefix and go to the judge, which (cached verdicts) accepted all of them: the
judge does not tell them apart either. The disagreement viewer does not list judge-task FPs (builder rule), so they
were not shown for review.

## Judge re-run with the current sim summary; judge-task false passes shown (2026-10-19)

The disagreement viewer now also lists labelled failures that pass the state checks of tasks ending in a judge, with
the judge's verdict and rationale on the card (62 cards; 10 in the unscored self-gen folder). All judge verdicts were
refreshed so the judge saw the current sim summary: 168 calls on labelled episodes (`judge_labeled.py`), 8 on policy
rollouts (`score.py --grader hybrid`). On the labelled failures that reach it the judge still says success on 40 of 62.
Judge on labelled episodes that passed the state checks: recall 0.921, specificity 0.338 (65 failures). Hybrid
(state checks + judge) on the 8 judge tasks: recall 0.921, specificity 0.721. Per task the judge rejects nothing on
00051, 00749, 00787, 01587, 01640 (specificity 0.000) and lets through only 52% of 01477 successes.
Policy hybrid SR: xwam 25.4%, motus 22.2% (was 22.6), gwp 20.0% (20.4), cosmos3 4.0% (4.7).

## Tool-use tasks graded by their own state verifiers, no vision judge (2026-10-19)

Why: the judge (gpt-6-luna, rubric r3) passed 40 of 62 labelled failures that reached it. Its rubric graded the best
moment ("success = accomplished at some time"), never mentioned unstable handling, and told it to trust the sim summary
("scrambled: yes") over the images, so dropped or thrown tools passed ("the spoon was later set down, but the stirring
had already been performed"). A bigger model alone (gpt-6.1-sol) caught 2 of 14 such failures; a stricter rubric (r4,
in score.py, not used for grading) caught 12-14 of 14 but also rejected 4-5 of 14 successes.

New: `tool_verifiers/` (one module per task, registered into verify.CHECKS; shared measurements in common.py):
tool-target contact points (which part of the tool touches which part of the target), the tool's working point in the
target's own frame (time inside, path, back-and-forth reversals), handling (regrasps, spin) and a shared ending
(not flung after the last release, target not knocked over / pushed away; per-task limits).
- 01640 drum (drum.py): stick touches the drumhead (top face) while held, >= 1 hit; any part of the stick (labelled
  successes often hit with the shaft); not flung > 1.0 m/s; drum not tipped > 45 deg.
- 00863 hammer (hammer.py): head (x >= 8.5 cm of the hammer mesh) strikes the box at >= 0.08 m/s; not flung > 1.2 m/s;
  the box may be knocked over (labelled successes tip it up to 130 deg).
- 00749 stir (stir.py): spoon bowl inside the pot while held >= 1 s, path >= 0.18 m; not flung > 1.2 m/s; pot tilt <= 100.
- 00787 scramble (scramble.py): blade inside the pan >= 1.2 s, path >= 0.2 m, >= 2 reversals of 2 cm; not flung > 1.0
  m/s; pan moved <= 0.4 m.
- 01587 scoop (scoop.py): spoon bowl in the plate region >= 0.2 s, path >= 0.07 m; not flung > 1.0 m/s; plate tilt <= 60.
- 00051 brush (brush.py): eraser held against the plate >= 0.3 s with >= 1 back-and-forth of 2 cm, in the plate's
  frame whatever its pose; not flung > 1.0 m/s.
- 01477 camera (camera_button.py): held up >= 1.9 s and not flung > 1.0 m/s. The sim has no button and the press is not
  visible in state (successes and failures touch the top face alike); fingertip time on the top face is reported only.

On the labels (recall / specificity): 01640 0.980 / 0.929, 00863 1.000 / 1.000, 00749 0.980 / 0.933, 00787 1.000 /
0.929, 01587 0.979 / 0.840, 00051 1.000 / 0.812, 01477 1.000 / 0.762. All 7: 0.991 / 0.884, against 0.921 / 0.721 for
state prefix + judge. All state-only tasks: n = 2906, recall 0.997, specificity 0.975.
Policy hybrid SR: xwam 26.2% (was 25.4), motus 21.1% (22.2), gwp 20.0% (20.0), cosmos3 3.6% (4.0).

## self-gen cup on the rack: state verifier (2026-10-19)

The spec said "rack not simulated" and ended in the judge, but the rack is in the scene: the static, collidable body
gigahands_stand_03. New `tool_verifiers/rack.py` (check `on_rack`): on the last 0.5 s the cup touches the rack, not the
table, is out of the hand and its lowest point is >= 3 cm above the table; not flung > 1.2 m/s after the last release.
Labels: recall 1.000, specificity 1.000 (50 / 11): every success ends hanging on the rack 6-13 cm up, every failure on
the table. Only 01242 still ends in a judge. State-only tasks: n = 2967, recall 0.997, specificity 0.976. Policy SR
unchanged (no policy rollout puts the cup on the rack).

## 00051 brush: plate face only, not dropped at the end (2026-10-19)

Reviewer: failure_b616bd26 "it does not brush, how is the brush logic written"; failure_4d0796d0 and failure_f961eb3d
"the brush dropped from hand at last".
- b616bd26 only ever touched the plate's rim (1.35 s, 0 s on its face); the rule counted any eraser-plate contact.
  `brush.py` `face_band_m`: contact points must lie on the plate's top or bottom face (within 1.2 cm of it, 1.5 cm in
  from the edge); back-and-forth strokes counted at 1 cm.
- 4d0796d0 / f961eb3d release the eraser 10-19 cm up and it falls slowly (0.17-0.77 m/s), under the not-flung limit.
  The brush stage now also takes the hand-based drop rule (`_not_dropped`, drop_rule hand_motion): still in hand at
  the end, or released low and deliberately.
00051: 1.000 / 0.812 -> 0.980 / 1.000. Lost: success_9f222d15 (eraser let go 3 cm up with fingers opening, but it
tumbles and ends tipped over, which the drop rule counts as a drop). Policy SR: motus 20.7% (was 21.1), others unchanged.

## Drop explanation: verdict first, each clause as measured vs limit (2026-10-19)

Reviewer: the "not dropped" text listed the whole rule and then the measurements, so it was hard to see which part
failed. It now starts with the verdict and names the clause that fired, e.g. "not dropped - DROPPED: tumbled after
release: turned 102 deg (limit 70 deg). Release: fingers opening (-0.40 rad/s; opening <= -0.1), hand level (+0.02
m/s; ...) -> deliberate release." Passing cases say why they pass ("ok: released only 0.005 m up (a drop counts from
0.02 m)"). Verdicts unchanged; all labelled recordings and policy rollouts re-verified.

## 00051 success_9f222d15: a low deliberate put-down that flops flat is not a drop; clearer drop clauses (2026-10-19)

Reviewer: "I still don't understand the reason it failed, it said deliberate release". The eraser was held on its edge,
set down 3 cm up with the fingers opening (deliberate), and then flopped flat (turned 102 deg). The tumble clause
counted even after a deliberate release. New `tumble_deliberate_min_m` (default 0 = unchanged): after a deliberate
release the tumble clause counts only when the release was at least this high. 00051 uses 5 cm: 9f222d15 (3 cm) passes,
failure_f961eb3d (deliberate, 10 cm, tumbles 121 deg) still fails. 00051: 1.000 / 1.000.
Each drop clause now states its own outcome: "SLIPPED OUT / LET GO TOO HIGH / TUMBLED / ENDED TIPPED OVER" when it
fires, otherwise "no (measured, limit)" or "over the limit but excused: deliberate release ...", and for a drop the
clause that fired comes first, the other checks after. Verdicts elsewhere unchanged (state-only 0.998 / 0.980).

## 00749 stir: the spoon must dip into the pot (2026-10-19)

Reviewer on failure_76f0789b: "it is never stirred, when the spoon head is in the pot, the hand already flew". The spoon
was held by the end of its handle with the bowl hovering at the pot's opening (the rule allowed the bowl centre up to
3 cm above the rim), then let go; the bowl only got into the pot after the release. The bowl is now tracked by its
lowest point (the tip of a tilted spoon) and must reach >= 6 cm below the rim while held: every labelled success reaches
7-15 cm, this episode 4.6 cm (other shallow failures 5.0-5.4 cm). 00749: 1.000 / 0.938 -> 1.000 / 1.000.
The stir / scramble / scoop explanation now lists each requirement with measured value and limit, failing ones first:
"stirred - FAILED: reaching 0.046 m below the rim (needs >= 0.06 m). ok: bowl inside ... 3.90 s (needs >= 1.0 s); ...".

## 00787 scramble: not dropped again and again while scrambling (2026-10-19)

Reviewer: the two remaining false passes (failure_bcbdcd1c, failure_fe990300) failed "because it is dropped".
New `max_drops_while_working` on the work checks: the hand-based drop rule is applied at every release between the
first and last frame of work (`verify._release_frames`, `_release_event(..., which=i)`), not only the last one.
Scramble allows at most 2 (labelled successes drop and pick up the spatula at most twice).
- bcbdcd1c drops it 3 times while scrambling (6.1 s, 6.4 s, 10.9 s) and now fails: "scrambled - FAILED: dropped 3
  times while working (at most 2)".
- fe990300 is not caught. It lets go three times (7.4, 12.2, 13.4 s) while the blade tip rests in the pan, so the handle
  falls 5-9 cm but the spatula's lowest point never leaves the pan (no drop by the rule); counting such handle falls
  does not separate it either (three labelled successes have two handle falls while scrambling as well).
00787: 1.000 / 0.929 -> 1.000 / 0.964.

## 01477 button dot, 01587 tray sent flying, 01640 light touch (2026-10-19)

- 01477 camera, reviewer: "there is a black round dot on top right of the camera, the finger should press that". The
  dot is drawn on the visual mesh at about (-1.7, +0.3, 3.3) cm in the camera frame (top-down render; successful
  fingertips cluster there). `camera_button.py` now needs a fingertip within 1.3 cm of it for >= 0.1 s while the camera
  is in hand. Catches failures 3fccb62e, 8ff7a93c, 99fc2ede. Fails four labelled successes whose fingertips never reach
  the dot (54992654 and e0eaa60a closest 2.9 cm, 93136ff9 3.3 cm, 58d76246 one frame): to review. failure_0581ddb8 and
  failure_46410afe do press the dot (0.35 s, 0.3 s) and still pass. 01477: 1.000 / 0.762 -> 0.920 / 0.905.
- 01587 scoop, failure_eef37b05 "the spoon presses the tray to suddenly fly away in the end": new `target_max_speed_m_s`
  in the shared ending (target not sent flying); scoop uses 1.5 m/s. Caught, no success lost. 01587: 1.000 / 0.958.
- 01640 drum, failure_d11597e0 "it does not actually hit the drum": the stick tip touches the middle of the drumhead
  twice (6.8 s, 7.7 s) at up to 0.51 m/s; six labelled successes strike softer (0.37-0.47 m/s), so neither contact place
  nor speed separates it. Unchanged.
State-only tasks: recall 0.997, specificity 0.990.

## 01477 camera: the button is pressed while the camera is steady (2026-10-19)

Reviewer on failure_0581ddb8: "the camera is just flying away, when the button is pressed the camera should be in a
rather steady state". The press now counts only on frames where the camera turns <= 2 rad/s, and the steady press must
last >= 0.2 s. 0581ddb8 presses the dot for 0.35 s but only 0.10 s of it steady (the camera spins at ~3.8 rad/s);
failure_46410afe is caught too. Lost: success_b07c7e6d (presses while the camera spins at ~4 rad/s: to review).
The four successes that never reach the dot were confirmed as failures by the reviewer. 01477: 0.978 / 1.000.
State-only tasks: recall 0.998, specificity 0.992. Policy SR unchanged.

## 01242 banana task skipped (2026-10-19)

The 22 labelled recordings of "Transfer the banana pieces into the salad bowl" were made on scene revision
jiangwei-2026-09-30, which has a free banana body (world_banana); the current benchmark scene has only the banana stand
and the bowl, so the recordings never matched the scene (not verified, not in the viewer) and policies were graded by the
judge on a task they cannot do. 01242 is now skipped like 00381 / 00636 until the banana is restored. No task is graded
by the vision judge any more. Policy SR over 270 decided rollouts: xwam 25.6%, motus 20.4%, gwp 20.0%, cosmos3 3.3%.

## Codex review fixes: work path, strokes, per-release drops (2026-10-19)

Codex review (20261019-0614) found three measurement errors; all fixed, thresholds unchanged.
1. The stir / scramble / scoop path counted travel between visits (outside the target). Now only steps between two
   consecutive working frames count (`common.work_in_target`).
2. Back-and-forth strokes were counted on all working frames concatenated, so the move from one visit to the next could
   count as a stroke, and the count was applied backwards in time. Now counted within each working session (visits less
   than 3 s apart; brushing is a series of short touches 1-1.5 s apart) and as they happen (`common.stroke_count`).
3. Counting drops at every release (scramble) judged each release against everything that happened later, including
   after a regrasp. Each release is now judged up to the next regrasp, and a release in the middle of the work counts as
   a drop only if it slipped out, was let go too high or tumbled (a spatula lying on its side in the pan between strokes
   is not a drop). The last-release rule used by every other task is unchanged.
Effect on all labelled recordings: one verdict changes. 00787 failure_bcbdcd1c passes again: judged correctly it drops
the spatula once while scrambling (10.85 s), as labelled successes do; the earlier count of three came from bug 3.
State-only tasks: recall 0.999, specificity 0.994. Policy SR unchanged.
New: VERIFIER_LOGIC.md (generated by _build_logic_doc.py) explains every rule with code pointers and a threshold
sensitivity audit (each tuned threshold at ±25%).
