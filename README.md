# ModernControls_v_Reinforcement Learning

These scripts are useful for comparing both controllers in such a way that:
    - Compare the controllers on the same paths
    - log each run into organized folders
    - generate a summary of results
    - generate figures

## Usage

In order to run the comparison, first one will need to select the parameters to be iterated through. Change the `SEED` to select a different sequence of 100 random tracks (it's deterministic) since we define the initial seed.

Copy the `models/` directory that was created from training with the `DDPG` repo into this folder. The `SEED_ID` is for selected the seed of the model you trained with. Make sure the `LABEL` matches the name of the checkpoint that was saved.

Next, simply uncomment the evaluation you want to do such as `wheelbase`, `hitch`, `velocity`, `sensor_noise`, and `control_frequency`. Be sure to only have one `PARAM_LABEL` and one `PARAMS`. The `PARAMS` list contains the changed parameter and the scripts will run it for the number of `DEMONSTRATIONS` defined.

```
SEED = 9
SEED_ID = [0]
LABEL = 'lp17_3_to_25'

#PARAM_LABEL = 'wheelbase'
#PARAMS = [8.192, 9.192, 10.192, 11.192, 12.192]

#PARAM_LABEL = 'hitch'
#PARAMS = [0.228, 0.114, 0.000, -0.114, -0.228]

#PARAM_LABEL = 'velocity'
#PARAMS = [-2.906, -2.459, -2.012, -1.564, -1.118]

#PARAM_LABEL = 'sensor_noise'
#PARAMS = [0.0, .03, .04, .05, .06]

#PARAM_LABEL = 'control_frequency'
#PARAMS = [.001, .010, .080, .400, .600]

PARAM_LABEL = 'test'
PARAMS = [0]

DEMONSTRATIONS = 100
```

Now run the comparison.
```
>>> python3 comparison.py
```

Alternatively, you can run the comparison on fixed tracks
```
>>> python3 comparison.py tracks.txt
```

## Rendering

Rendering does not occur by default because it speeds up testing time. Simply type into the CLI the following and press enter. It does not matter that things print out, just type it fast enough and press enter.

```
>>> render
```

If you no longer want to look at rendering, simply type the following again

```
>>> hide
```

## What to expect

After `comparison.py` is done, a folder with the `PARAM_LABEL` will be created where a folder for each `PARAMS`, which contain folders `run0`, `run1`, `run2` etc. A `run<#>` folder will contain two files: 1) MCTrue.txt and 2) RLFalse.txt. But really, it is labeled as such to quickly determine if the run reached the goal or not. These files contain the iteration log for that run.

A summary file will be created that will likely be named something similar to `stat_me_<PARAM_LABEL>_<VALUE>.txt`. This contains the metric values averaged over each of the runs.  
```
│
└───wheelbase
    ├───10_192
        ├───run0
            mcFalse.txt
            rlFalse.txt
        ├───run1
            mcFalse.txt
            rlFalse.txt
        ├───run2
            mcFalse.txt
            rlFalse.txt
   ├───stat_me_wheelbase_10_192.txt
```

## Do stats

In order to generate bar graphs from the `stat_me_<PARAM_LABEL>_<VALUE>.txt` files, run the following command

```
>>> python3 do_stats.py <PARAM_LABEL>
```

## Do rm_stats

Considering the LQR jackknifes, this skews the data so we filter for only the runs where both make it to the goal.

```
>>> python3 do_rm_stats.py <PARAM_LABEL>
```

## Generate error plots, singular bar charts, and 3D surface plots

```
>>> python3 error.py <PARAM_LABEL>/<VALUE>/run<>
```

## Generate report for summary table over all `PARAMS`

- Goal 
- Fin
- Jackknife
- Error Too Large
- Angle Too Large
- Out of Bounds

```
>>> python3 report_stats.py <PARAM_LABEL>
```