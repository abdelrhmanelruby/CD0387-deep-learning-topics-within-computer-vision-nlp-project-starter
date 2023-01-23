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
Debugging and profiling were carried out in accordance with the instructions provided throughout the course :
```python
rules = [
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
]
```

### Results
![image](https://user-images.githubusercontent.com/107134115/214050037-cf64097b-a741-4b16-8ee8-db9c86a00d8b.png)




## Model Deployment
To deploy the model, an extra file called model.py was required
This script loads the model and transforms the input.
```python
pytorch_model = PyTorchModel(model_data=estimator.model_data, 
                             role=role, 
                             entry_point='model.py', 
                             py_version='py36',
                             framework_version='1.8')


predictor = pytorch_model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
```
Screenshot of the endpoint in operation
![Screenshot_20230123_072357](https://user-images.githubusercontent.com/107134115/214050510-50f04561-6242-4b8e-93f8-500a00183d33.png)

