#!/bin/sh

# from JTunis's <https://github.com/JTunis/create-pytorch-lambda-layer>


# parse inputs
for arg in "$@"
do
    case $arg in
        --use-docker=*) USE_DOCKER="${arg#*=}" shift;;
        --folder=*) FOLDER="${arg#*=}" shift;;
        --python=*) PYTHON_VERSION="${arg#*=}" shift;;
        --torch=*) TORCH_VERSION="${arg#*=}" shift;;
        --torchvision=*) TORCHVISION_VERSION="${arg#*=}" shift;;
        --torchaudio=*) TORCHAUDIO_VERSION="${arg#*=}" shift;;
        *) echo "ERROR: Invalid argument ${arg}" && exit 1;;
    esac
done

case ${USE_DOCKER} in
    yes) ;;
    no) ;;
    "") echo "ERROR: No folder specified" && exit 1;;
    *) echo "ERROR: Invalid docker option. Expected values: yes/no" && exit 1;;
esac

case ${FOLDER} in
    "") echo "ERROR: No folder specified" && exit 1;;
esac

# derive c_python_version from input python version
case ${PYTHON_VERSION} in
    2.7) C_PYTHON_VERSION="cp27";;
    3.5) C_PYTHON_VERSION="cp35";;
    3.6) C_PYTHON_VERSION="cp36";;
    3.7) C_PYTHON_VERSION="cp37";;
    3.8) C_PYTHON_VERSION="cp38";;
    3.9) C_PYTHON_VERSION="cp39";;
    "") echo "ERROR: No Python version specified" && exit 1;;
    *) echo "ERROR: Invalid Python version. Expected versions include: 2.7, 3.5, 3.6, 3.7, 3.8, 3.9" && exit 1;;
esac

# derive flagged_c_python_version from input python version
# these include the "m" flag in the version to show it was implemented with Pymalloc
# the "m" flag was removed in Python 3.8: https://docs.python.org/3/whatsnew/3.8.html#build-and-c-api-changes
case ${PYTHON_VERSION} in
    2.7) FLAGGED_C_PYTHON_VERSION="cp27m";;
    3.5) FLAGGED_C_PYTHON_VERSION="cp35m";;
    3.6) FLAGGED_C_PYTHON_VERSION="cp36m";;
    3.7) FLAGGED_C_PYTHON_VERSION="cp37m";;
    3.8) FLAGGED_C_PYTHON_VERSION="cp38";;
    3.9) FLAGGED_C_PYTHON_VERSION="cp39";;
    "") echo "ERROR: No Python version specified" && exit 1;;
esac

# build torch wheel URL
case ${TORCH_VERSION} in
    "") "ERROR: No Torch version specified" && exit 1;;
    *) TORCH_WHEEL="https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-${C_PYTHON_VERSION}-${FLAGGED_C_PYTHON_VERSION}-linux_x86_64.whl";;
esac

# build torchvision wheel URL
case ${TORCHVISION_VERSION} in
    "") TORCHVISION_WHEEL="";;
    *) TORCHVISION_WHEEL="https://download.pytorch.org/whl/cpu/torchvision-${TORCHVISION_VERSION}%2Bcpu-${C_PYTHON_VERSION}-${FLAGGED_C_PYTHON_VERSION}-linux_x86_64.whl";;
esac

# build torchaudio wheel URL
case ${TORCHAUDIO_VERSION} in
    "") TORCHAUDIO_WHEEL="";;
    *) TORCHAUDIO_WHEEL="https://download.pytorch.org/whl/torchaudio-${TORCHAUDIO_VERSION}-${C_PYTHON_VERSION}-${FLAGGED_C_PYTHON_VERSION}-linux_x86_64.whl";;
esac

# cleanup function to run if the script errors
failure_cleanup()
{
    echo "Cleaning up due to failure..."
    rm -rf $FOLDER 
    exit 1
}


echo Destination folder: $FOLDER
echo Python $PYTHON_VERSION
echo use docker? $USE_DOCKER
echo torch $TORCH_VERSION : ${TORCH_WHEEL}
echo torchvision $TORCHVISION_VERSION : $TORCHVISION_WHEEL
echo torchaudio $TORCH_WTORCHAUDIO_VERSION : $TORCH_WTORCHAUDIO_WHEELHEEL

# create directory into which packages will be installed
mkdir -p $FOLDER


if [[ $USE_DOCKER == no ]]
then
    
    echo "Building Python packages ..."
    
    pip$PYTHON_VERSION install \
    ${TORCH_WHEEL} \
    ${TORCHVISION_WHEEL} \
    ${TORCHAUDIO_WHEEL} \
    -t $FOLDER --no-cache-dir || failure_cleanup

else
    echo "Building Python packages with Docker..."
    docker run -v "$PWD":/var/task "lambci/lambda:build-python${PYTHON_VERSION}" /bin/sh -c \
    "pip$PYTHON_VERSION install \
    ${TORCH_WHEEL} \
    ${TORCHVISION_WHEEL} \
    ${TORCHAUDIO_WHEEL} \
    -t $FOLDER; exit" || failure_cleanup
fi


# remove extraneous files and directories
echo "Removing extraneous files/directories...\n"
cd "$FOLDER" || failure_cleanup
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf ./{caffe2,wheel,wheel-*,pkg_resources,boto*,aws*,pip,pip-*,pipenv,setuptools}
rm -rf ./{*.egg-info,*.dist-info}
find . -name \*.pyc -delete

echo "Crated ${FOLDER}."
