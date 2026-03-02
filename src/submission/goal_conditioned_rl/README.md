## Using Goal Conditioned RL Environment

## Run the Code

To run the goal conditioned RL code, invoke the following command from the submission directory:

```
python -m goal_conditioned_rl.main -- env <ENV> -- num_bits=<BIT_NUM> -- num_epochs=<EPOCHS> --
her_type <HER_TYPE>
```

Where
- `ENV` is `sawyer-reach` or `bit-flip`
- `BIT_NUM` is number of bits 6,15, etc
- `EPOCHS` are total number of epochs (250, 500, etc)
- `HER_TYPE` is the HER strategy which can be one of `no_hindsight`, `final`, `random`, and `future`

This will create a directory `logs/gcrl`, which will contain saved tensor files from each experiment run determined by the parameters above.