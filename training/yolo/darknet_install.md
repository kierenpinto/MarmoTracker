How I installed Darknet with CUDA.

1.) Install CUDA: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

2.) Install GCC Version less than 8.

sudo apt install gcc-7 g++-7

3.) Create a symlink to gcc version 7. This is so NVCC uses GCC v7. Not compatible with versions >8.

sudo ln -s /usr/bin/gcc-7 /usr/local/cuda/bin/gcc

4.) git clone https://github.com/pjreddie/darknet.git

4.) cd darknet

6.) Update the Makefile.

GPU=1

CC=gcc-7
CPP=g++-7

7.) Ensure CUDA has environment variable so that nvcc (compiler for CUDA) works.

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin

8.) make

9.) Further details: https://pjreddie.com/darknet/install/
