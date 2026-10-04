FAVOR supplementary video
==========================

File: favor_supp.mp4 (H.264, 960x1032, 20 fps, 105 s, no audio)
Plays in any standard video player (VLC, QuickTime, Windows Media Player, web browsers).

Contents
- Title and the four LIBERO tasks with the Franka Panda.
- Locked-joint scenarios, each shown side by side for four methods (B1, W-IK pos, W-IK pose,
  Priority IK) in real time. Every panel uses the same seed and the same diffusion samples;
  only the execution-time correction differs. The link attached to the locked joint is red,
  the end-effector path is overlaid, and a badge marks success or failure.
  - Bowl-Stove, joint 7 locked, seed 10001
  - Soup, joint 3 locked, seed 10000
  - Bowl-Stove, joint 1 locked, seed 10001
  - Soup, joint 2 locked, seed 10000 (kinematically unrecoverable: no method succeeds)

All shown episodes are re-rendered from the evaluation sweep and reproduce its recorded
outcomes. Code: https://github.com/lloydkwak/FAVOR
