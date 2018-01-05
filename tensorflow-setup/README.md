# Tensorflow Setup:

### Refer to this guide:
https://www.tensorflow.org/install/

### You should make sure you have the correct NVidia drivers installed:
- CUDA Drivers - http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows
- CUDNN - CUDA for Deep Neural Networks - https://developer.nvidia.com/cudnn

### Set up virtual environment for CPU Tensorflow and GPU Tensorflow
```
conda create --name tensorflow python=3.5
activate tensorflow
conda install jupyter
conda install scipy
pip install tensorflow
```

```
conda create --name tensorflow-gpu python=3.5
activate tensorflow-gpu
conda install jupyter
conda install scipy
pip install tensorflow-gpu
```

### To activate any virtual environment:
```
activate tensorflow
activate tensorflow-gpu
```

### Additional Notes
- On Windows, Tensorflow 1.4 is only compatible with CUDA 8.0, not CUDA 9.X
- Use check_version.py to check if all tensorflow components have been installed correctly.


##### List avialable GPU's
```
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
```
