# Setting up the lambda function

## Steps

1. Create a Security Group SG  
    Inbound Rule: Custom TCP, Port range: `2049`, Source: anywhere

2. Create the EFS file system in some VPC & subnet(s).  
    Add the SG created above.

3. Create Access Point for lambda to access EFS:  
    - User ID: 1000  
    - Group ID: 1000  
    - Permissions: 777  
    - Root directory path: `/lambda`

4. Create EC2 instance on the same VPC (minimum t2.small):

   1. Setup the instance packages:

        ``` bash
        sudo yum -y update
        sudo reboot
        ```

   2. Mount EFS on `~/efs`  
    Follow <https://docs.aws.amazon.com/efs/latest/ug/wt1-test.html>  
    or click on 'attach' on EFS console.

   3. Install Python3.8

        ``` bash
        sudo amazon-linux-extras install python3.8
        # makes python 3 default
        sudo rm /usr/bin/python3
        sudo ln -s /usr/bin/python3.8 /usr/bin/python3

        ```

   4. pip install packages on `~/efs/lambda/pkgs`  

        ``` bash
        pip3 install --update --taget ~/efs/lambda/pkgs --no-chche-dir torch==1.11.0 torchvision==0.12.0
        ```

   5. Save the model in `~/efs/lambda/models`  

        ``` python
        model = torchvision.models.vgg16(num_classes=10) # for CIFAR10
        torch.save(model, '~/efs/lambda/models/vgg16.pt')
        ```

   6. Save the dataset in `~/efs/lambda/datasets`:  

        ``` python
        torchvision.datasets.CIFAR10(
            root='~/efs/lambda/datasets',
            train=True,
            download=True
        )
        ```

   7. Make the directory for lambda workers gradient results:  

        ```bash
        mkdir ~/efs/lambda/runs
        ```

5. Create IAM-Role for lambda function:  
    - AWSLambdaExecute  
    - AWSLambdaVPCAccessExecutionRole  
    - AmazonElasticFileSystemClientFullAccess

6. Create lambda function:  
   - Use the same VPC/Subnets
   - Use the same SG
   - Assign the IAM-Role
   - Add the EFS and Access point, root dir: `/mnt/lambda`
   - Copy the file `app.py` (`app.lambda_handler` is the handler)

### References

- Useful tutorial series on EFS Lambda integration:  
<https://www.youtube.com/playlist?list=PL5KTLzN85O4L0rYTtGVKxPr4yQ5oHMYOn>

- AWS walkthrough for EFS + Lambda:  
<https://docs.aws.amazon.com/lambda/latest/dg/configuration-filesystem.html>
