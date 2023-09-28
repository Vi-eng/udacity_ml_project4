# Udacity_ml_project4
The fourth project in the ML Fundamentals Course with Udacity

**NOTEBOOK USED: [starter.ipynb](https://github.com/Vi-eng/udacity_ml_project4/blob/master/starter.ipynb)**

In this project, I trained a model that classifies images of vehicles in order to assign scone delivery locations (near for bicycles and far for motorbikes) using Amazon Sagemaker.

I deployed the model as an endpoint and then made calls to the model using [three lambda functions.](https://github.com/Vi-eng/udacity_ml_project4/blob/master/lambda.py)
I then organized my workflow using [step functions](https://github.com/Vi-eng/udacity_ml_project4/blob/master/StepfunctionDef.py).
The [execution results](https://github.com/Vi-eng/udacity_ml_project4/blob/master/execution-detail.json) are illustrated below:

![stepfunctions_graph_fail](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/f59d3b95-7a28-4be1-855b-3049ba3843cc)
![stepfunctions_graph](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/9ff55cd1-6b3f-4423-b34c-d00bf5d2e549)

The last lambda function is a check to ensure the function performs above a specified threshold.
 
After which I visualized the predictions made.
![download](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/af3c8e98-f97c-4b3a-ba7c-e0996d9b835d)

**NOTEBOOK USED: [starter-Optional.ipynb](https://github.com/Vi-eng/udacity_ml_project4/blob/master/starter-Optional.ipynb)**

I also tried to expand on the concepts learnt in the course of the project to train another model which classified more vehicles (multiclass classification) from the CIFAR-dataset used.
I also attempted parallel processing in my [lambda functions](https://github.com/Vi-eng/udacity_ml_project4/blob/master/lambda-opt.py) using batch transform (I was able to launch the transforms successfully but require more knowledge on how to process payload for image classification batch transform models)
I modified the [step function](https://github.com/Vi-eng/udacity_ml_project4/blob/master/execution-detail-BJ.json) to fit these new lambda functions.

![stepfunctions_graph_flow1](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/8d25e344-ae35-4a17-8492-66120ad114aa)
![stepfunctions_graph_flow2-BJ](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/85a3a547-84d8-4518-a71f-ac89a04dddb3)

I also tried to visualize the results of the multiclass model.
![download](https://github.com/Vi-eng/udacity_ml_project4/assets/68657918/9a21a56c-5fe1-492f-9cf9-a2ad61ce0739)
