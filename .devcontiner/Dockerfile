FROM  pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    x11-apps

RUN conda install opencv
RUN conda install matplotlib

RUN conda init bash
RUN echo "conda activate base" > ~/.bashrc
# RUN echo "source ~/.bashrc && conda activate base" >> ~/.bashrc
RUN python -m pip install --upgrade pip
RUN python -m pip install soundfile einops librosa tensorboard trimesh pyvista pykeops