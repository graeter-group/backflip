TORCH_VERSION=${1:-"2.6.0"}
CUDA_VERSION=${2:-"124"}
PYTHON_VERSION=${3:-"3.12"}
BACKFLIP_DIR=${4:-"backflip"}

if [ ! -d "$BACKFLIP_DIR" ]; then
	echo "Error: Backflip directory '$BACKFLIP_DIR' does not exist."
	exit 1
fi

SUPPORTED_VERSIONS=("3.10" "3.12" "None")
# assert that python version is either 3.10, 3.12 or None:
if [[ ! " ${SUPPORTED_VERSIONS[*]} " =~ " ${PYTHON_VERSION} " ]]; then
    echo "Error: Unsupported python version $PYTHON_VERSION. Supported versions are: ${SUPPORTED_VERSIONS[*]}"
    exit 1
fi

echo "Installing backflip with torch $TORCH_VERSION and cuda $CUDA_VERSION for python $PYTHON_VERSION in directory $BACKFLIP_DIR"

# wait for 3 seconds in case the user wishes to cancel the installation
sleep 3

THISDIR=$(dirname "$(readlink -f "$0")")

pushd "${THISDIR}/../.."

# install torch
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# add the torch version to the requirements file to make sure it is not overwritten
if [[ "$PYTHON_VERSION" != "None" ]]; then
    REQUIREMENTS_FILE="requirements_${PYTHON_VERSION//./}.txt"
else
    REQUIREMENTS_FILE="requirements.txt"
fi

cp $BACKFLIP_DIR/install_utils/$REQUIREMENTS_FILE $BACKFLIP_DIR/install_utils/tmp_requirements.txt
echo -e "\ntorch==$TORCH_VERSION" >> $BACKFLIP_DIR/install_utils/tmp_requirements.txt

# install pypi dependencies:
pip install -r $BACKFLIP_DIR/install_utils/tmp_requirements.txt

rm $BACKFLIP_DIR/install_utils/tmp_requirements.txt

# install gafl from source:
# Note: this is a temporary solution until gafl is available on pypi
git clone https://github.com/hits-mli/gafl.git
pushd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
popd

# Finally, install backflip:
cd $BACKFLIP_DIR
pip install -e . # Install backflip

popd
