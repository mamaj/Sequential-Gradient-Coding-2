#!/bin/sh

# REGION="ca-central-1"
# REGION="ap-southeast-2"
REGION="ap-northeast-1"
# REGION="eu-west-2"

SOURCE_BUCKET="s3://torch-layers"
DEST_BUCKET="s3://torch-layers-${REGION}"
OBJECT="torch1.11.0-py3.8-vgg"

# creates bucket
# aws s3 mb \
#     $DEST_BUCKET \
#     --region $REGION


# uploads runtime zip file
# aws s3 cp $SOURCE_BUCKET/$OBJECT \
    # $DEST_BUCKET \

# update the function
aws lambda update-function-code \
    --function-name ${REGION}:506827543107:function:torch-vgg16-2 \
    --zip-file fileb://deployment_package/package.zip \
    --region $REGION
