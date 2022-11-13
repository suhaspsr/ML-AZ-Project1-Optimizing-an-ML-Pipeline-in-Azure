# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The objective of this project is to create and optimize an ML pipeline using Azure ML SDK. We are using Scikit-learn Logistic Regression model to do hyperparameter tunning using HyperDrive and then we will compare the results of tunned logistic regression model with optimized AutoML model. We are dealing with classification problem where we are trying to predict whether customer will participate in bank services.

The AutoML VotingEnsemble method has slightly better accuracy at 91.75% compared to hyperparameter tunned Logistic Regression model with 91.21%. AutoML has 0.5% better performance than the tunned LR model. This is not a big difference in performance.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
There are three important steps in designing for hyperparameter tunning. 

###Sampling Methods:

Sampling methods for parameters of the model. We can to do random sampling, grid search or use Bayesian sampling. Choose the parameters which you want to tune for depending on the model. For each parameter provide the search space depending on continuous or discreate values.

For this project we are choosing Regularization value and maximum number of iterations for LR model. 
```
parameter = { "--C" : choice(0.001,0.01, 0.1, 1, 10, 100, 200, 1000), "--max_iter" : choice(10, 50, 100, 200, 500, 1000) }
```
We are using RandomParameterSampling because its simpler and computation cost is small compared to other methods.

###Early Stopping Policy:

We need to use a stopping criterion for sampling methods to automatically terminate poorly performing jobs to be computationally efficient.

We are using BanditPolicy as our stopping criteria with evaluation interval at 2 and slack factor equals 0.1. The slack will compare the current run with the best performed run to be in a minimum threshold and determines whether to terminate or not.

###Performance Metric:

We need to provide a performance metric to the configuration to be maximized or minimized. We are using accuracy as our metric here and tune the parameters to get the best accuracy.


## AutoML
Based on the classification task AutoML checkes for best performing models and provides the model with highest accuracy and its parameters.  
```
AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=data,
    label_column_name='y',
    n_cross_validations=5)
```

We need to provide parameters into run AutoML for stopping criteria, what kind of task “classification” or “Regression”, metric to maximize or minimize etc. AutoML searched all classification algorithms like Logistic Regression, SVM, Random Forrest, LightGBM, and XGBoost with its variation along with Ensemble Voting method.


## Pipeline comparison
The accuracy between the two models is very small around 0.5%. So there is no clear winner in a way. The VotingEnsemble and Logistic Regression are two different algorithms. In voting, it is using top performing algorithms with its various hyperparameters to predict and the using voting to give you final prediction whereas you are using one hyperparameters tunned Logistic Regression model for prediction. AutoML is very easy for developers as it does every task automatically for you with data validation. With hyperdrive you need to design and run the experiments with its variations. Its manual and laborious process but as a developer we have 100% control. 

## Future work
As a Data Scientist, its important to do exploratory analysis on the data. There is no scope for that in this project. So, as a future work I would like to do EDA for better understanding of the data and interpretability. 

I want to use different parameter sampler method such as Grid Search or Bayesian sampling and different parameters to fine tune. Based on the EDA and objective I want explore different algorithms and metrics to maximize or minimize. 

## Proof of cluster clean up

