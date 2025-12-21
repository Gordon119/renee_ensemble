conda create -y -n renee python=3.10

conda activate renee

# CUDA toolkit (provides nvcc for building apex)
conda install -c nvidia/label/cuda-12.8.0 cuda -y

# PyTorch built against CUDA 12.8 runtime (adds Blackwell sm_120 support)
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

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

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# git clone https://github.com/NVIDIA/apex
# cd apex
# git checkout 25.09


# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..
