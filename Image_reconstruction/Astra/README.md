Astra toolbox is a GPU based Matlab and Python toolbox for 2D and 3D tomography reconstruction. 
Source: https://www.astra-toolbox.com/

Traditionally, CT reconstruction relied on back projection, which is limited by the Nyquist sampling frequency. Algebraic reconstruction technique (ART) is an effective technique to combat this limitation and thus achieve a reduction in CT radiation dose and increase of time resolution.

In this example, I showed how to use advanced CT reconstruction technique and new sampling method to increase the time resolution of dynamic CT. The result is published on SPIE [Developments in X-ray Tomography103910M, 2017, San Diego, California, United States](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10391/103910M/Micro-CT-in-situ-study-of-carbonate-rock-microstructural-evolution/10.1117/12.2273877.full?SSO=1). The 8 subset (on the left) was resampled and reconstructed from the original dataset (on the right). Through the proposed method, the time resolution was increased by 8-fold. The raw projection data can be downloaded in https://sciencedata.dk/shared/proj7. The data was acquired at the 3D Imaging Center at Denmark. The data was also used in [Journal of Hydrology
Volume 571, April 2019, Pages 21-35](https://www.sciencedirect.com/science/article/pii/S0022169419300988) with model simulations to explain how the increase in surface area correlates with the size of the Damköhler space.

<img src=example.png height = 500>


## Prerequirements
Installation of Astra tool box by downloading and unzipping the toolbox matlab code and add its path in the enviroment

