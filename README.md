# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Here's a screen shot of some completed training jobs:
![image](https://user-images.githubusercontent.com/107134115/213970240-86da24c0-b160-47af-9a17-88d277883173.png)

The following is a list of hyperparameters that have been tuned:
"lr": ContinuousParameter(0.001, 0.1),
"batch-size": CategoricalParameter([32, 64, 128, 256]),
"epochs": IntegerParameter(2, 5)

Here's a screen shot of Logs metrics :
![image](https://user-images.githubusercontent.com/107134115/213970682-57dc328c-9981-4316-b389-57a4bde1c60a.png)
![image](https://user-images.githubusercontent.com/107134115/213970704-25447ae4-188e-40e7-8888-ffc48de375c4.png)

The best best hyperparameters:
'batch-size': '"64"',
'epochs': '3',
'lr': '0.00716803840042681'

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
