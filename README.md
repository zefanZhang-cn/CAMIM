# Source Code of CAMIM for Multimodal Relation Extraction
Official Implementation of our Paper "Caption-Aware Multimodal Relation Extraction with Mutual Information Maximization" (Authors: Zefan Zhang, Weiqi Zhang, Yanhui Li, Bai Tian, indicates equal contribution) in ACM MM 2024.
## Motivation
.<img src="Figure/first.png" width="450" height="450" /> 

Previous methods tend to introduce the issue of ***error sensitivity*** and be easily affected by ***irrelevant object information*** from the image, such as the person in the red box. Therefore, we try to leverage ***detailed captions of entities*** in a given image, which can eliminate the influence of irrelevant objects and improve the efficiency of relation extraction.
## Model Architecture
.<img src="Figure/model.png" width="1800" height="450" /> 

The framework of the proposed Caption-Aware MultiModal Relation Extraction Network with Mutual Information Maximization (CAMIM). (a) We utilize the Multimodal Large Language Model to extract captions and encode them by BERT, leverage ResNet50 to encode the image, and translation image \cite{rethinking} to get object-level features \cite{HVPChen}. (b) In the Caption-Aware Module, we hierarchically aggregate features from different levels and feed them into the Cross-attention Module to interact. To reduce the risk of overfitting, we feed outputs and text features together into the Fusion Module. (c) We finally leverage Mutual Information to preserve the crucial information.

## Required Environment
To run the codes, you need to install the requirements for [RE](requirements.txt).

    pip install -r requirements.txt

## Data Preparation
* MNRE
  
  You need to download three kinds of data to run the code.  
  > 1.The raw images of [MNRE](https://github.com/thecharm/MNRE), many thanks.  
  > 2.The visual objects from the raw images from [HVPNeT](), many thanks.  
  > 3.The generated images of [MNRE]() from TMR, many thanks.
  > 4.Our  generated [Captions]().
  Then you should put folders ``img_org``,  ``img_vg``,  ``diffusion_pic``,  ``caption``  under the "./data" path.
