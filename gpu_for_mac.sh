# 建立 conda 環境
conda create -n torch-mps python=3.10
conda activate torch-mps

# 安裝 Metal 支援的 PyTorch 版本
conda install pytorch torchvision torchaudio -c pytorch-nightly