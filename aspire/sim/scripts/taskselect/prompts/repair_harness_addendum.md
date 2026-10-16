
HARNESS ADDITIONS (they come after the instructions above and extend them):
1. A read-only diagnosis tool, CPU only. Run it on a directory of trials, for example PROBE_DIR/candidate_A/eval or
   TASK_DIR/iter_NN/candidate_A/eval (several directories at once are fine):
     .venv/bin/python3 scripts/taskselect/harness/why.py <dir> [<dir> ...]
   Per trial it prints the failure stage and tags, every SAM3 text query (best score, runner-up score, pixel of the best mask),
   the centre of every fitted box, and every gripper close with the IK target before it. At the end it counts, per query, the
   trials where the object was not seen, and compares where passing and failing trials grasped. Use it before you write an
   iteration: a query that is often unseen, two near-tied detections, or a grasp target that differs between passing and
   failing trials each point at a cause. In findings.md name the why.py output that supports every root cause you claim.
2. Your kept program will be tested later against the program you started from, on seeds you cannot see and that are not
   51-80. A change that only helps on 51-65 is rejected and its lesson is not promoted. Prefer fixes that rest on a
   diagnosed cause over fixes found by trying settings.
