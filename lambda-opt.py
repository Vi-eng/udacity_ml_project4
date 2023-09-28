import json
import boto3
import base64
import random

def generate_test_case(n = 1):
    s3_str = ''
    keys_list = []
    bucket = 'project4-vehicles'
    
    # Setup s3 in boto3
    s3 = boto3.resource('s3')

    # Randomly pick from sfn or test folders in our bucket
    objects = s3.Bucket(bucket).objects.filter(Prefix = "test")

    # Grab any random object key from that folder!
    for i in range(n):
        obj = random.choice([x.key for x in objects])
        keys_list.append(obj) 
        # s3_str += f'{dict(keys_list)} \n'
        
    return {
        "image_data": "",
        "s3_bucket": bucket,
        "s3_key": keys_list
    }


s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    internal_event = generate_test_case(event['n'])

    key = internal_event['s3_key']## TODO: fill in
    bucket = internal_event['s3_bucket'] ## TODO: fill in
    
    image_data = []
    n = event['n']
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    
    # We read the data from a file
    for i in range(n):
        filename = '/tmp/image.png'
        s3.download_file(bucket, key[i], filename)
        
        with open(f"/tmp/image.png", "rb") as f:
            image_data.append(base64.b64encode(f.read()))
            
        if n > 1:
            s3.upload_file('/tmp/image.png', bucket, f'sample_imgs/image{i}.png')

    # Pass the data back to the Step Function
    print("Event:", internal_event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": [],
            "n": n,
            'job': event['job'],
            "samples_path": f"s3://{bucket}/sample_imgs"
        }
    }

#LAMBDA 2
import json
import boto3
import base64

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-09-28-22-43-12-950' 
#session= session.Session()
runtime = boto3.Session().client('sagemaker-runtime')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'][0])
    #envoke the endpoint using boto3 runtime
    response = runtime.invoke_endpoint(EndpointName = ENDPOINT, ContentType = 'image/png', Body = image)
    predictions = json.loads(response['Body'].read().decode())
    # We return the data back to the Step Function #Create a new key to the event named   "inferences"  
    event['body']["inferences"] = predictions
    #print(json.dumps(event))
    
    return {
        'statusCode': 200,
        'body': event['body']
        }

#LAMBDA 3	
# import json


# THRESHOLD = .7


# def lambda_handler(event, context):
    
#     # Grab the inferences from the event
#     inferences = event["body"]["inferences"] ## TODO: fill in
    
#     # Check if any values in our inferences are above THRESHOLD
#     meets_threshold = (max(inferences) >= THRESHOLD)  ## TODO: fill in
    
#     # If our threshold is met, pass our data back out of the
#     # Step Function, else, end the Step Function with an error
#     if meets_threshold:
#         pass
#     else:
#         raise("THRESHOLD_CONFIDENCE_NOT_MET")

#     return {
#         'statusCode': 200,
#         'body': json.dumps(event['body'])
#     }


import json
import boto3
import base64
# import matplotlib.pyplot as plt

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-09-28-11-19-19-034' 
#session= session.Session()
runtime = boto3.Session().client('sagemaker-runtime')

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('boto')

def lambda_handler(event, context):
    # Decode all the image data
    # image = []
    bucket = event['body']['s3_bucket']
    # s3 = boto3.resource('s3')
    
    # for i in range(len(event['body']['image_data'])):
    #     image = base64.b64decode(event['body']['image_data'][i])
    #     plt.imsave(f'image.png', image)
        
    #     s3_client = boto3.client('s3')
    #     s3_client.upload_file('image.png', bucket, f'img_sample/image{i}.png')
    
        # file_name =  f'img_sample/image{i}.png'
        # file_content = image
         
        # object = s3.Object('bucket', file_name)
        # object.put(Body=file_content)
        
    #invoke batch transform model predictions
    client = boto3.client('sagemaker')
    
    model = client.model.Model(
        image_uri = client.image_uris.retrieve(framework = 'image-classification', region = 'us-east-1'), ## TODO: fill in
        model_data = 's3://project4-vehicles/models/image_model/image-classification-2023-09-28-10-59-38-512/output/model.tar.gz',
        role = 'arn:aws:iam::023941299171:role/service-role/AmazonSageMaker-ExecutionRole-20230513T234630'
)
    transformer = model.transformer(
    instance_count=1, 
    instance_type='ml.m4.xlarge', 
    output_path=f's3://{bucket}/batch_output'
)
    transformer.transform(
    data=event['body']['samples_path'], 
    data_type='S3Prefix',
    content_type='images/png', 
    split_type='Line'
)

    
    # transformer = client.create_transform_job(
    #     TransformJobName=event['body']['job'],
    #     ModelName='image-classification-2023-09-28-11-19-19-034',
    #     TransformInput={
    #         'DataSource': {
    #             'S3DataSource': {
    #             'S3DataType': 'S3Prefix',
    #             'S3Uri': event['body']['samples_path']
    #          }
    #      },
    #     'SplitType': 'Line'
    #     },
    #     TransformOutput={
    #         'S3OutputPath': f's3://{bucket}/batch_output',
    #         'AssembleWith': 'Line'
    #     },
    #     TransformResources={
    #     'InstanceType': 'ml.m4.xlarge',
    #     'InstanceCount': 1})
        
    # #envoke the endpoint using boto3 runtime
    # response = runtime.invoke_endpoint(EndpointName = ENDPOINT, ContentType = 'image/png', Body = image)
    # predictions = json.loads(response['Body'].read().decode())
    # # We return the data back to the Step Function #Create a new key to the event named   "inferences"  
    # event['body']["inferences"] = predictions
    # #print(json.dumps(event))
    
    return {
        'statusCode': 200,
        'body': event['body']
        }