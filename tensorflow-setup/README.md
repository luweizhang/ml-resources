# Tensorflow Setup:

### Refer to this guide:
https://www.tensorflow.org/install/


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
