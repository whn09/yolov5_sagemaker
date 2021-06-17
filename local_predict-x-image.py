# locally
# docker run -v $(pwd)/model:/opt/ml/model -p 8080:8080 yolov5 serve
import requests
import json
from io import BytesIO
url='http://localhost:8080/invocations'
image_uri = 'data/images/inference/bus.jpg'
img = BytesIO(open(image_uri, 'rb').read())
payload = img
r = requests.post(url,data=payload,headers={'content-type': 'application/x-image'})
print(r.json())

# on sagemaker
# python create_endpoint
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
    image_uri = input_image
    img = BytesIO(open(image_uri, 'rb').read())
    payload = img

    sagemaker_runtime_client = boto3.client('sagemaker-runtime', config=config)
    session = Session(sagemaker_runtime_client)

#     runtime = session.client("runtime.sagemaker",config=config)
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName='paddleocr',
        ContentType="application/x-image",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)

infer('data/images/inference/bus.jpg')