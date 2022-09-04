import argparse
import subprocess
import sys
from pathlib import Path
import zipfile


def create_package(
    python_version='3.8',
    package_dir='./package',
    lab_dir='pkgs',
    model_path='models/vgg16.pt',
    dataset_dir='datasets/',
    runs_dir='runs/' ,
    efs_zip_name='efs.zip',
    dataset_name='CIFAR10',
    num_classes=10,
    use_docker=True,
):
    
    package_dir = Path(package_dir)    
    model_path = Path(model_path)
    
    package_dir.mkdir(exist_ok=True)

    # pkgs
    subprocess.run([
        './make-pkgs.sh',
        f'--use-docker={"yes" if use_docker else "no"}',
        f'--folder={package_dir/lab_dir}',
        f'--python={python_version}',
        '--torch=1.11.0',
        '--torchvision=0.12.0'
        ],
    )

    sys.path.append(f'{package_dir/lab_dir}')
    import torch
    import torch.nn as nn
    import torchvision

    # models
    (package_dir / model_path).parent.mkdir(exist_ok=True)
    
    if model_path.stem == 'cnn': 
        m = nn.Sequential(
            nn.Conv2d(1, 8, 5), nn.ReLU(),
            nn.Conv2d(8, 16, 5), nn.ReLU(),
            nn.Conv2d(16, 4, 5), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    elif model_path.stem == 'vgg16':
        m = torchvision.models.vgg16(num_classes=num_classes)
        
    torch.save(m, package_dir / model_path)

    # datasets
    (package_dir / dataset_dir).mkdir(exist_ok=True)
    dataset = torchvision.datasets.__getattribute__(dataset_name)(
        root=package_dir / dataset_dir,
        train=True,
        download=True,
    )
    for p in (package_dir / dataset_dir).glob('*.tar.gz'):
        p.unlink()

    # runs
    Path(package_dir / runs_dir).mkdir(exist_ok=True)

    # zip
    if efs_zip_name and efs_zip_name != 'na':
        print('zipping the package...')
        with zipfile.ZipFile(efs_zip_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in package_dir.rglob("*"): #TODO: needs checking 
                zip_file.write(file, file.relative_to(package_dir))

    print('Created the EFS package.')
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--python-version', type=str, required=True)
    parser.add_argument('--package-dir', type=str, required=True)
    parser.add_argument('--lab-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--runs-dir', type=str, required=True)
    parser.add_argument('--efs-zip-name', type=str, required=False)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--use-docker', action='store_true', default=False)

    args = parser.parse_args()
    
    create_package(**vars(args))