conda create -y -n renee python=3.8

source activate
conda activate renee

conda install cuda -c nvidia/label/cuda-11.8.0 -y

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install transformers
pip install scipy
pip install pandas
pip install cython
pip install scikit-learn
pip install sentence-transformers
pip install seaborn
pip install numpy
pip install numba

git clone https://github.com/kunaldahiya/pyxclib
cd pyxclib
pip install .
cd ..

git clone https://github.com/NVIDIA/apex 
cd apex
git checkout 5c9625cfed681d4c96a0ca4406ea6b1b08c78164
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
