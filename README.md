# Stochastic Gradient Push on Ray
## Goal
Run SGP code with Ray, with minimal change from original code.

## Things changed from SGP code
- nvidia apex related code removed (was not used in paper)

## Files
- ray_runner.py, ray_trainer.py, ray_sgp_util.py

## What works
- Running training steps.

## TODO
- After training, gather output logfile('{tag}_r{}_n{}.csv') from worker nodes, and plot results.
- Restore checkpoint by get/set_state()
- Rather than generating logfile on each worker node, ray_trainer gathers the stats and writes logfile.
- Proper interfacing (need to discuss)
- (more things to be added)
