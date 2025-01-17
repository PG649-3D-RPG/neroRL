# Evaluate a model

To evaluate a series of model checkpoints utilize the `python eval_checkpoints.py` command.

```
"""
    Usage:
        evaluate.py [options]
        evaluate.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --path=<path>              Specifies the tag of the tensorboard summaries [default: None].
        --name=<path>              Specifies the full path to save the output file [default: results.res].
"""
```

## --config

In general, [training](training.md), evaluating and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/otc.yaml` argument to specify your configuration file.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --path
This is the path to the directory that contains the checkpoints.

## --name
This is the file path to save the results of the evaluation.

## Example
```
python eval_checkpoints.py --config=./configs/minigrid.yaml --path=./checkpoints/default/20200527-111513_2 --name=results.res
```
This command
- loads all checkpoints located in the stated directory,
- evaluates these on the Minigrid environment and
- outputs the file results.res with the evaluation's results.

## Plotting the results
In order to work with the results check out this [jupyter notebook](../notebooks/plot_checkpoint_results.ipynb).
