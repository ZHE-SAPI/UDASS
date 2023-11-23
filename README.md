# DA-STC
The official repo for "DA-STC: Domain Adaptive Video Semantic Segmentation via Spatio-Temporal Consistency".    
The model and test code is unloaded, and the full code will be uploaded soon.  
[arxiv paper](https://arxiv.org/abs/2311.13254)    


# Abstract
Video semantic segmentation is a pivotal aspect of video representation learning. However, significant domain shifts present a challenge in effectively learning invariant spatio-temporal features across the labeled source domain and unlabeled target domain for video semantic segmentation. To solve the challenge, we propose a novel DA-STC method for domain adaptive video semantic segmentation, which incorporates a bidirectional multi-level spatio-temporal fusion module and a category-aware spatio-temporal feature alignment module to facilitate consistent learning for domain-invariant features. Firstly, we perform bidirectional spatio-temporal fusion at the image sequence level and shallow feature level, leading to the construction of two fused intermediate video domains. This prompts the video semantic segmentation model to consistently learn spatio-temporal features of shared patch sequences which are influenced by domain-specific contexts, thereby mitigating the feature gap between the source and target domain. Secondly, we propose a category-aware feature alignment module to promote the consistency of spatio-temporal features, facilitating adaptation to the target domain. Specifically, we adaptively aggregate the domain-specific deep features of each category along spatio-temporal dimensions, which are further constrained to achieve cross-domain intra-class feature alignment and inter-class feature separation. Extensive experiments demonstrate the effectiveness of our method, which achieves state-of-the-art mIOUs on multiple challenging benchmarks. Furthermore, we extend the proposed DA-STC to the image domain, where it also exhibits superior performance for domain adaptive semantic segmentation. The source code and models will be made available at [DA-STC](https://github.com/ZHE-SAPI/DA-STC).

Index Terms: Domain adaptation, video semantic segmentation, feature alignment, spatio-temporal consistency.
  

# A short video demo of our results:
https://github.com/ZHE-SAPI/DA-STC/assets/52643313/d5738a66-8e77-4718-a851-4218d500f800


# Installation
1. create conda environment  
    `conda create -n TPS python=3.6`  
    `conda activate TPS`  
    `conda install -c menpo opencv`  
    `pip install torch==1.2.0 torchvision==0.4.0`  
2.clone the ADVENT repo  
    ·git clone https://github.com/valeoai/ADVENT`  
    `pip install -e ./ADVENT`  
3. clone the current repo   
    `git clone https://github.com/ZHE-SAPI/DA-STC.git`  
    `pip install -r ./DASTC/requirements.txt`  
4. resample2d dependency  
    `cd \DASTC\dastc\utils\resample2d_package`  
    `python setup.py build`  
    `python setup.py install`   

# Data Preparation  
please refer to the structure of the folder \video_seg\DASTC\data  

# Pretrained Models  
Download here and put them under  DASTC\pretrained_models.  

# Optical Flow Estimation  
pleasr first refer to [https://github.com/xing0047/TPS#optical-flow-estimation](https://github.com/Dayan-Guan/DA-VSN/issues/1), the full optical data will be unloaded soon.  

# Train and Test  
1. Train  
    
`cd /DASTC`  
`python ./dastc/scripts/train_DAVSS_DSF_cd_ablation_syn.py --cfg ./dastc/scripts/configs/dastc_syn2city.yml`   
`python ./dastc/scripts/train_DAVSS_DSF_cd_ablation_viper.py --cfg ./dastc/scripts/configs/dastc_viper2city.yml`    

(train code will be uploaded soon.)  

2. Test  

`cd /DASTC`  
`python ./dastc/scripts/test_DAVSS_DSF_cd_ablation_syn.py --cfg ./dastc/scripts/configs/dastc_syn2city.yml`  
`python ./dastc/scripts/test_DAVSS_DSF_cd_ablation_viper.py --cfg ./dastc/scripts/configs/dastc_viper2city.yml`  
 
# Acknowledgement  
This codebase is heavily borrowed from [https://github.com/Dayan-Guan/DA-VSN], [https://github.com/GaoLii/DSP], [https://github.com/microsoft/ProDA/tree/main].  
