## OffDetector, a DeepFake Detector

## Abstract

Deepfake techniques have made it increasingly easy to replace and manipulate a person's face within a video. The underlying technology, generative adversarial networks (GANs), are trained to produce realistic images from random seed noise. Deepfakes present an increasingly large cultural impact on trust in the legitimacy of online videos. Utilizing a large novel data set of Deepfake videos provided by AWS, Facebook and Microsoft through a Kaggle competition, we trained a binary classifier - the OffDetector - to detect facial manipulations. Using state of the art facial detection, efficient use of data batching and GPU technologies, our classifier achieves an ROC AUC of 0.973 on an unseen dataset of manipulated and original videos, the FaceForensics dataset, with an inference time of 4 seconds per video.

## Authors
- <a href="https://github.com/adamwlev">@adamwlev</a>
- <a href="https://github.com/diegomontoyas">@diegomontoyas</a>
- <a href="https://github.com/TheSuri">@TheSuri</a>
- <a href="https://github.com/KieranCTaylor">@KieranCTaylor</a>
- <a href="https://github.com/pabloherrera8994">@pabloherrera8994</a>
- <a href="https://github.com/https://github.com/RyanGale-AK">@RyanGale-AK</a>

## Credit for Face Detection
https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices
