import json
import boto3
import base64
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


import boto3
import json
import base64

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2021-12-13-14-15-50-236"
runtime= boto3.client('runtime.sagemaker')
s3 = boto3.client('s3')

sagemaker = boto3.client('sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    with open("/tmp/imageToSave.png", "wb") as fh:
        fh.write(base64.b64decode(event["image_data"]))

    # Make a prediction:
    with open("/tmp/imageToSave.png", "rb") as f:
        payload = f.read()
        payload = bytearray(payload)
        
    inferences = runtime.invoke_endpoint(
        EndpointName=ENDPOINT, ContentType="application/x-image", Body=payload
    )
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences['Body'].read().decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


import json
import re


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    event = json.loads(event["body"])
    inference1 = json.loads(event["inferences"])[0]
    inference2 = json.loads(event["inferences"])[1]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (THRESHOLD < inference1) or (THRESHOLD < inference2)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
