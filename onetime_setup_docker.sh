
apt-get update && apt-get install -y curl


# Install miniconda to /miniconda
curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
rm Miniconda-latest-Linux-x86_64.sh
conda update -y conda


conda install -y \
    scikit-image \
    flask \
    pillow

