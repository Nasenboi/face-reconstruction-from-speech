# Acoustic-to-Anthropometric Feature Mapping

This code is the practical implementation of my master thesis "The Acoustic-to-Anthropometric Feature Mapping: Correlating Speech Parameters with Facial Geometry". The goal of the thesis is to find out which features of a persons voice correlate in what way with the appearance of a persons face.
To find such features, first of all a list is congreated, which contains auditory features which may correlate to anthropometric measurements (AMs) of a persons face. Thereafter, a machine learning algorithm is implemented to predict the AMs using the vector of gathered audio features. 
Lastly, explainablitly algorithms will uncover the algorithmic process of the models prediction, revealing features, that were relevant for local decisions. By gathering and plotting multiple local feature imporances global patterns shall be discovered.

# Preable

## Wiki

The wiki contains relevant infromations to understand the architecture of the code at hand. Make sure to read it after this readme to get a better understanding of the project.

## Prequisites

**! The code is only tested on Linux so far !**  
#### Hardware:
- [Nvidia and CUDA-able GPU](https://developer.nvidia.com/cuda-gpus)
- Enough Harddrive space for a large Dataset for training (Depending on the Dataset)
- Generally a decent PC would be advantagous

#### Software:
- [cuda](https://developer.nvidia.com/cuda-toolkit)
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [ffmpeg](https://www.ffmpeg.org/)
- [anaconda or miniconda](https://anaconda.org/)
- [docker](https://www.docker.com/)

# Credits

This project would not be possible without the code and models developed by other people, below are honorable mentions:

## Rethinking Voice-Face Correlation: A Geometry View

[GitHub](https://github.com/lxa9867/VAF)

## Deep3DFaceRecon PyTorch

[Github](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

## Face Alignment
[GitHub](https://github.com/1adrianb/face-alignment)

## NVDIFFRAST

[GitHub](https://github.com/NVlabs/nvdiffrast)

## Insightface

[GitHub](https://github.com/deepinsight/insightface)

## Basel Face Model

P. Paysan, R. Knothe, B. Amberg, S. Romdhani and T. Vetter, "A 3D Face Model for Pose and Illumination Invariant Face Recognition," 2009 Sixth IEEE International Conference on Advanced Video and Signal Based Surveillance, Genova, Italy, 2009, pp. 296-301, doi: 10.1109/AVSS.2009.58. keywords: {Lighting;Face recognition;Face detection;Shape;Image sensors;Power generation;Computer vision;Costs;Image analysis;Image reconstruction;Basel Face Model (BFM);Morphable Model;generative 3D face models;statistical models;database;2D/3D fitting;recognition;identification},