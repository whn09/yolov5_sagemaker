import boto3
from botocore.config import Config
from boto3.session import Session
import json

config = Config(
    read_timeout=120,
    retries={
        'max_attempts': 0
    }
)

def infer(input_image):
    bucket = 'sagemaker-cn-north-1-813110655017'
    image_uri = input_image
    test_data = {
        'bucket' : bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)
    print(payload)

    sagemaker_runtime_client = boto3.client('sagemaker-runtime', config=config)
    session = Session(sagemaker_runtime_client)

#     runtime = session.client("runtime.sagemaker",config=config)
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName='yolov5',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)
    return result

def lambda_handler(event, context):
    # TODO implement
    result = infer('data/images/inference/bus.jpg')
    return {
        'statusCode': 200,
        'body': json.dumps(result, ensure_ascii=False)
    }
