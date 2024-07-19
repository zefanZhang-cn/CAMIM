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
  > 2.The visual objects from the raw images from [HVPNeT](https://github.com/zjunlp/HVPNeT), many thanks.  
  > 3.The generated images from [TMR](https://github.com/thecharm/TMR), many thanks.
  > 4.Our  generated [Captions]().
  
  Then you should put folders ``img_org``,  ``img_vg``,  ``diffusion_pic``,  ``caption``  under the "./data" path.

## Path Structure
The expected structures of Paths are:  
### Multimodal Relation Extraction
```
CAMIM
 |-- ckpt # save the checkpoint
 |-- data
 |    |-- txt  # text data
 |    |    |-- ours_train.txt # input data
 |    |    |-- ours_val.txt
 |    |    |-- ours_test.txt
 |    |    |-- mre_train_dict.pth  # {imgname: [object-image]}
 |    |    |-- ...
 |    |    |-- dif_train_weight_strong.txt  # strong correlation score for generated image
 |    |    |-- train_weight_strong.txt  # strong correlation score for original image
 |    |    |-- dif_train_weight_weak.txt  # weak correlation score for generated image
 |    |    |-- train_weight_weak.txt  # weak correlation score for original image
 |    |    |-- ...
 |    |    |-- phrase_text_train.json # {imgname: phrase for object detection}
 |    |    |-- ...
 |    |    |-- mre_dif_train_dif.pth # {imgname: [coordinates]}
 |    |    |-- ...
 |    |-- img_org       # original image data, please download it according to the above link
 |    |-- img_vg   # visual object image data for original image, please download it according to the above link
 |    |-- diffusion_pic   # generated image data, please download it according to the above link
 |    |-- caption   # Qwen caption data
 |    |    |-- BLIP_train.txt ... # BLIP caption data
 |    |    |-- caption_train.txt ... # Qwen caption data
 |    |    |-- cogvlm_train.txt ... # Cogvlm caption data
 |    |    |-- instruct_train.txt ... # Instruct-BLIP caption data
 |    |    |-- llavanext_train.txt ... # Llavanext caption data
 |    |    |-- minicpm_train.txt ... # minicpm caption data
 |    |-- ours_rel2id.json # target relations
 |-- opennre	# main framework 
 |    |-- encoder # main model
 |    |    |-- bert_encoder.py # TMR-RE
 |    |    |-- modeling_bert.py
 |    |-- framework # processing files
 |    |    |-- data_loader.py # data processor
 |    |    |-- sentence_re.py # trainer
 |    |    |-- utils.py
 |    |-- model # classifier
 |    |    |-- softmax_nn.py # main classifier
 |    |    |-- modeling_bert.py 
 |    |    |-- base_model.py # supporting the classifier, no modification required
 |    |-- tokenization # tokenizers, no modification required
 |    |-- pretrain.py # basic file
 |    |-- utils.py # basic file
 |-- opennre.egg-info
 |-- run.py   # main 
```
`bert_model.py` is the file for our CAMIM model.

`data_loader.py` is the file for processing raw data.

`sentence_re.py` is the file that sets up training, testing, and other processes.

`run.py` is used for running the whole program.

## Citation

If you find this repo helpful, please cite the following:

``` latex

```
