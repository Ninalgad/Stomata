# Stomata - Pump Failure Detection using XGBoost
`Username: jackson6` \
Source code to a top 25 solution to the [Helios - Pump Failure Detection Data Science Challenge](https://www.topcoder.com/challenges/21731d2d-8647-466f-a485-b7edd03b5e2e).
The proposed solution is the average of 15 xgboost models built on hand-crafted features derived from the time-series data. Here, we vectorize time series data using various statistics using information taken at most one week before the inference time. The models directly output the seven-dimensional probability vector where XGBoost models each probability separately. Here, we optimize the training hyperparameters using the Optuna package.

## Submission format
The submission contains following content:

```
/solution
   solution.csv
/code
   Dockerfile
   data.py              # data processing
   modeling.py          # model training scripts
   requirements.txt     # required packages
   test.py
   test.sh
   train.py
   train.sh
   utils.py             # utilities
   loc_ohe.pkl          # one-hot-encoding object used in data processing
   motor_ohe.pkl
   pump_ohe.pkl
```

### Docker Usage
Ideally, we require an accessible GPU accelerated runtime to train our models. \
An example command to build gpu docker image:
```
nvidia-docker build -t jackson6 .
```


### train.sh

The training script will delete the old model files, create the training dataset (in about 3 hours), and train the models. Note the training data will not be saved. \
An example sample call to the training script:
```
./train.sh /content/train/
```

Here we assume that the training data looks like this:
```
content/
    train/
        operations-data/
            ...
        run-life/
            ...
        equipment_metadata.csv
```

### test.sh

An example sample call to the test script:
```
./test.sh /content/test/ /content/solution.csv
```

Here we assume that the training data looks like this:
```
content/
    test/
        operations-data/
            ...
        run-life/
            ...
        equipment_metadata.csv
```
where the output will be saved to /content/solution.csv in this example.


# References
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://dl.acm.org/doi/10.1145/3292500.3330701)
