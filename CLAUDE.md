All code should live in a git environment in the current working directory.

If a git repository does not already exist, create one with the name of the current directory, and push it to github.

All code and artifact changes should be pushed to github. Commit often. Most likely this should be done after any logical code change, and likely after each command you run that finishes writing code. The only thing that should go in the codebase is ML data and large models. 

All models should go into a results/checkpoints directory. In the same directory should be a json tracking all relevant metrics for each experiment and checkpoints (for example, val metric, test metric). Charts can go in results/charts. 

There should also be a results/experiments.json file which tracks the commit we ran the experiment with and a thorough textual overview of the experiment as well as the exact training command used to run the experiment.

When writing training code, always make sure it's resumable from a checkpoint, and always make sure best val and final checkpoints are produced.

When running remotely, those checkpoints should always be pulled locally. Ideally, they should also be synced to some external storage (git is fine, maybe we'll find something better along the way). 

When running on a remote server like runpod, frequently ping to check the status of the experiment in case it failed, and when it fails, try to fix it and take necessary steps to continue.

Running on the remote pod should always happen via github syncs and scripts to make sure we're not relying on scp'ing commands.
