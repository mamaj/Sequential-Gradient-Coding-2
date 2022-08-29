import json
import boto3

region_name = 'Sydney'
size = 50
batch = 1


with open('regions2.json') as f:
    REGIONS = json.load(f)
    region = REGIONS[region_name]

session = boto3.session.Session()
client = session.client('lambda', region_name=region['code'])    

event = {'size': size, 'batch': batch}
r = client.invoke(
    FunctionName=region['arn'],
    InvocationType='RequestResponse',
    LogType='Tail',
    Payload=json.dumps(event)
)

resp = json.loads(r['Payload'].read().decode())
print(resp)