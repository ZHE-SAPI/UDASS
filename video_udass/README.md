### Unified Domain Adaptive Semantic Segmentation (Video)

## 🔍 Main Results

#### 🔁 SYNTHIA-Seq -> Cityscapes-Seq

|          Methods          |        road        |       side.       |       buil.       |        pole        |       light       |        sign        |       vege.       |        sky        |        per.        |       rider       |        car        |        mIoU        |
| :------------------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|           Source           |        56.3        |        26.6        |        75.6        |        25.5        |        5.7        |        15.6        |        71.0        |        58.5        |        41.7        |        17.1        |        27.9        |        38.3        |
|           DA-VSN           |        89.4        |        31.0        |        77.4        |        26.1        |        9.1        |        20.4        |        75.4        |        74.6        |        42.9        |        16.1        |        82.4        |        49.5        |
|          PixMatch          |        90.2        |        49.9        |        75.1        |        23.1        |        17.4        |        34.2        |        67.1        |        49.9        |        55.8        |        14.0        |        84.3        |        51.0        |
|           I2VDA           |        89.9        |        40.5        |        77.6        |        27.3        |        18.7        |        23.6        |        76.1        |        76.3        |        48.5        |        22.4        |        82.1        |        53.0        |
|            TPS            |        91.2        |        53.7        |        74.9        |        24.6        |        17.9        |        39.3        |        68.1        |        59.7        |        57.2        |        20.3        |        84.5        |        53.8        |
|            SFC            |        90.9        |        32.5        |        76.8        |        28.6        |        6.0        |        36.7        |        76.0        |        78.9        |        51.7        |        13.8        |        85.6        |        52.5        |
|          TPL-SFC          |        90.0        |        32.8        |        80.4        |        28.9        |        14.9        |        35.3        |        80.8        |        81.1        |        57.5        |        19.6        |        86.7        |        55.3        |
|            PAT            |        91.5        |        41.3        |        76.1        |        29.6        |        20.9        |        33.8        |        72.4        |        75.9        |        51.3        |        24.7        |        86.2        |        54.9        |
|            CMOM            |        90.4        |        39.2        |        82.3        |        30.2        |        16.3        |        29.6        |        83.2        |        84.9        |        59.3        |        19.7        |        84.3        |        56.3        |
| ***QuadMix(CNN)*** |      *90.8*      |      *39.9*      | ***83.2*** |      *33.2*      |      *30.1*      |      *50.7*      |      *84.8*      |      *83.2*      |      *61.2*      |      *32.7*      |      *87.4*      |      *61.5*      |
| ***QuadMix(ViT)*** | ***94.1*** | ***61.9*** |      *82.9*      | ***36.9*** | ***41.0*** | ***59.1*** | ***85.2*** | ***85.6*** | ***64.3*** | ***37.8*** | ***90.3*** | ***67.2*** |

#### 🔁 VIPER -> Cityscapes-Seq

|          Methods          |        road        |       side.       |       buil.       |       fence       |       light       |        sign        |       vege.       |       terr.       |        sky        |        per.        |        car        |       truck       |        bus        |       motor       |        bike        |        mIoU        |
| :------------------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|           Source           |        56.7        |        18.7        |        78.7        |        6.0        |        22.0        |        15.6        |        81.6        |        18.3        |        80.4        |        59.9        |        66.3        |        4.5        |        16.8        |        20.4        |        10.3        |        37.1        |
|          PixMatch          |        79.4        |        26.1        |        84.6        |        16.6        |        28.7        |        23.0        |        85.0        |        30.1        |        83.7        |        58.6        |        75.8        |        34.2        |        45.7        |        16.6        |        12.4        |        46.7        |
|           DA-VSN           |        86.8        |        36.7        |        83.5        |        22.9        |        30.2        |        27.7        |        83.6        |        26.7        |        80.3        |        60.0        |        79.1        |        20.3        |        47.2        |        21.2        |        11.4        |        47.8        |
|            TPS            |        82.4        |        36.9        |        79.5        |        9.0        |        26.3        |        29.4        |        78.5        |        28.2        |        81.8        |        61.2        |        80.2        |        39.8        |        40.3        |        28.5        |        31.7        |        48.9        |
|            MoDA            |        72.2        |        25.9        |        80.9        |        18.3        |        24.6        |        21.1        |        79.1        |        23.2        |        78.3        |        68.7        |        84.1        |        43.2        |        49.5        |        28.8        |        38.6        |        49.1        |
|           I2VDA           |        84.8        |        36.1        |        84.0        |        28.0        |        36.5        |        36.0        |        85.9        |        32.5        |        74.0        |        63.2        |        81.9        |        33.0        |        51.8        |        39.9        |        0.1        |        51.2        |
|            SFC            |        89.9        |        40.8        |        83.8        |        6.8        |        34.4        |        25.0        |        85.1        |        34.3        |        84.1        |        62.6        |        82.1        |        35.3        |        47.1        |        23.2        |        31.3        |        51.1        |
|          TPL-SFC          |        89.9        |        41.5        |        84.0        |        7.0        |        36.5        |        27.1        |        85.6        |        33.7        |        86.6        |        62.4        |        82.6        |        36.3        |        47.6        |        23.2        |        31.9        |        51.7        |
|            PAT            |        85.3        |        42.3        |        82.5        |        25.5        |        33.7        |        36.1        |        86.6        |        32.8        |        84.9        |        61.5        |        83.3        |        34.9        |        46.9        |        29.3        |        29.9        |        53.0        |
|            CMOM            |        89.0        |        53.8        |        86.8        |        31.0        |        32.5        |        47.3        |        85.6        |        25.1        |        80.4        |        65.1        |        79.3        |        21.6        |        43.4        |        25.7        |        40.6        |        53.8        |
| ***QuadMix(CNN)*** | ***91.6*** | ***51.4*** |      *87.0*      |      *24.1*      |      *32.3*      | ***37.2*** |      *84.1*      | ***28.4*** |      *84.8*      |      *64.4*      |      *85.7*      |      *41.4*      |      *46.5*      |      *34.0*      |      *49.6*      |      *56.2*      |
| ***QuadMix(ViT)*** |      *87.3*      |      *43.8*      | ***87.3*** | ***25.2*** | ***40.0*** |      *36.9*      | ***86.7*** |      *20.8*      | ***90.3*** | ***65.8*** | ***86.8*** | ***48.6*** | ***65.6*** | ***37.6*** | ***49.7*** | ***58.2*** |

Note: We are the first to explore transformer network in the video unsupervised domain adaptation for semantic segmentation.

## ⚙️ Environment Setup

##### **A unified environment applicable to both image and video UDASS.**

1. create conda environment

```bash
conda create -n UDAVSS_py38 python=3.8 -y
conda activate UDAVSS_py38
conda install -c menpo opencv
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

2. clone the [ADVENT repo](https://leftgithub.com/valeoai/ADVENT)

```bash
git clone https://github.com/valeoai/ADVENT
pip install -e ./VIDEO/ADVENT
```

3. clone the current repo

```bash
pip install -r ./requirements.txt
```

4. resample2d dependency:

```
cd ./video_udass/VIDEO/tps/utils/resample2d_package
python setup.py build
python setup.py install
```

5. install mmcv-full, this command compiles mmcv locally and may take some time

```shell
pip install mmcv-full==1.3.7  # requires other packeges to be installed first
```

## 📁 Data Preparation

1. [Cityscapes-Seq](https://www.cityscapes-dataset.com/)

```
VIDEO/data/Cityscapes/
VIDEO/data/Cityscapes/leftImg8bit_sequence/
VIDEO/data/Cityscapes/gtFine/
```

2. [VIPER](https://playing-for-bencVhmarks.org/download/)

```
VIDEO/data/Viper/
VIDEO/data/Viper/train/img/
VIDEO/data/Viper/train/cls/
```

3. [Synthia-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar)

```
VIDEO/data/SynthiaSeq/
VIDEO/data/SynthiaSeq/SEQS-04-DAWN/
```

## 📁 Optical Flow Estimation

✅ The steps to generate the optical flow (recommended✅) (refer to [issue](https://github.com/Dayan-Guan/DA-VSN/issues/1)):

1. git clone git clone -b sdcnet [https://github.com/NVIDIA/semantic-segmentation.git](https://github.com/NVIDIA/semantic-segmentation.git);
2. Unzip the files of [video_udass/Code_for_Optical_Flow_Estimation.zip](https://github.com/ZHE-SAPI/UDASS/blob/main/video_udass/Code_for_Optical_Flow_Estimation.zip) and put them in the folder of sdcnet;
3. Run the shell scripts to generate optical flow:
   ```
   [1] CItyscapes validation set: "python Cityscapes_val_optical_flow_scale512.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/Cityscapes --target_dir Cityscapes_val_optical_flow_scale512 --vis --resize 0.5"

   [2] SynthiaSeq train set: "python Estimated_optical_flow_SynthiaSeq_train.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/SynthiaSeq/SEQS-04-DAWN/rgb --target_dir Estimated_optical_flow_SynthiaSeq_train --vis --resize 0.533333"

   [3] Viper train set: "python Estimated_optical_flow_Viper_train.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/viper --target_dir /home/dayan/gdy/adv/snapshots/Estimated_optical_flow_Viper_train--vis --resize 0.533333"
   ```

✅ You can also download the estimated optical flow of all datasets here, and put them as subfolders in the [./VIDEO/data](https://github.com/ZHE-SAPI/UDASS/tree/main/video_udass/VIDEO/data) folder.

|            Dataset            |  Folder Name  |                           Download Link                           |                      Relative Path                      |
| :----------------------------: | :------------: | :---------------------------------------------------------------: | :------------------------------------------------------: |
|  **Synthia-Seq_train**  | train_folder 1 | [Download](https://pan.baidu.com/s/1pWuMpJBkLUMetjzIx9dbKA?pwd=unif) |   TPS/tps/data/estimated_optical_flow_synthiaseq_train   |
|     **VIPER_train**     | train_folder 1 | [Download](https://pan.baidu.com/s/1IedArcO6OW7fXzs4NvFPIw?pwd=unif) |     TPS/tps/data/estimated_optical_flow_viper_train     |
|                                | train_folder 2 | [Download](https://pan.baidu.com/s/1DPSDZJytSJYvmlr4SksRKA?pwd=unif) |                                                          |
|                                | train_folder 3 | [Download](https://pan.baidu.com/s/1xbkKml5tn1Bvmzskue1pLQ?pwd=unif) |                                                          |
|                                | train_folder 4 | [Download](https://pan.baidu.com/s/1PLZfMKwCNxr65SAnCQbSQw?pwd=unif) |                                                          |
|                                | train_folder 5 | [Download](https://pan.baidu.com/s/1gsTDkKa3unAy5jsfxJR86w?pwd=unif) |                                                          |
| **Cityscapes-Seq_train** | train_folder 1 | [Download](https://pan.baidu.com/s/1SQZp6bqXJih9hBFeDO2fjA?pwd=unif) | TPS/tps/data/estimated_optical_flow_cityscapes_seq_train |
|                                | train_folder 2 | [Download](https://pan.baidu.com/s/19rIzZ6KYyo5KR_ikGSOfGA?pwd=unif) |                                                          |
|                                | train_folder 3 | [Download](https://pan.baidu.com/s/1y2XYYJW8MNY0RceZBOvviA?pwd=unif) |                                                          |
|  **Cityscapes-Seq_val**  |   val folder   | [Download](https://pan.baidu.com/s/10JBF43JeFMFjGSr5e8ittw?pwd=unif) |  TPS/tps/data/estimated_optical_flow_cityscapes_seq_val  |

Please merge different `train_folder` parts of the same dataset (e.g., `train_folder 1`, `train_folder 2`, etc.) into a single directory for consistency and ease of access.

## 🔧 Dataset and Optical Flow Path Update Instructions

Please update the dataset paths in the `.yml` files located in `/video_udass/VIDEO/tps/scripts_ablation/configs/`.

Additionally, modify the paths in the `scripts_ablation` to match your actual project directory structure.

## 🏋️ Train and Test

- ✅ Train (the core code will be available soon.)

```
  cd /video_udass/VIDEO (Please adjust according to the actual address of your device manually )
  # syn2city CNN
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_24.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # viper2city CNN
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_31.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
  # syn2city ViT
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_24_former.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # tps_viper2city ViT
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_31_former.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
```

- ✅ Test (can in parallel with Train) (run the test code to validate the results.)

```
  cd /video_udass/VIDEO
  # syn2city CNN
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_24.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # viper2city CNN
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_31.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
  # syn2city ViT
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_24_former.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # viper2city ViT
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_31_former.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
```

## 💾 Checkpoints

Below, we provide checkpoints of UDASS for the different benchmarks.

```
VIDEO/pretrained_models
```

|           Benchmark           | Model Type |                                        Download Link                                        |          Location          |
| :---------------------------: | :--------: | :-----------------------------------------------------------------------------------------: | :-------------------------: |
| Synthia-Seq → Cityscapes-Seq |    ViT    | [Download](https://drive.google.com/file/d/1kwzpghUD1UiK6AvQyazSw0gMGYjAUCwe/view?usp=sharing) | `VIDEO/pretrained_models` |
|    VIPER → Cityscapes-Seq    |    ViT    | [Download](https://drive.google.com/file/d/1OCDnHlz2lJplnPcV7iINOhiRLUTeNm6P/view?usp=sharing) | `VIDEO/pretrained_models` |
| Synthia-Seq → Cityscapes-Seq |    CNN    | [Download](https://drive.google.com/file/d/1XJ5naWs9wuZ8k1r6VRHx6YRF7gK5HuCV/view?usp=sharing) | `VIDEO/pretrained_models` |
|    VIPER → Cityscapes-Seq    |    CNN    | [Download](https://drive.google.com/file/d/1TGpysDaBkQ3F-NQj9wTL0JJnclQMQmto/view?usp=sharing) | `VIDEO/pretrained_models` |

## Acknowledgement

This codebase is is based on the following open-source projects. We thank their authors for making the source code publicly available.

* [TPS](https://github.com/xing0047/TPS/tree/main)
* [DA-VSN](https://github.com/Dayan-Guan/DA-VSN)
* [Sdcnet ](https://github.com/NVIDIA/semantic-segmentation)
* [FlowNet 2](https://github.com/NVIDIA/flownet2-pytorch)

## 📚 Citation

If you find **UDASS** helpful in your research, please consider giving us a ⭐ on GitHub and citing our work in your publications!

```bibtex
@ARTICLE{10972076,
  author={Zhang, Zhe and Wu, Gaochang and Zhang, Jing and Zhu, Xiatian and Tao, Dacheng and Chai, Tianyou},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Unified Domain Adaptive Semantic Segmentation},
  year={2025},
  volume={47},
  number={8},
  pages={6731-6748},
  keywords={Videos;Semantics;Optical flow;Training;Adaptation models;Transformers;Optical mixing;Artificial intelligence;Semantic segmentation;Minimization;Unsupervised domain adaptation;semantic segmentation;unified adaptation;domain mixup},
  doi={10.1109/TPAMI.2025.3562999}}
```
