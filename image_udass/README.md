# Unified Domain Adaptive Semantic Segmentation (Image )

## 🔍 Main Results

### 🔁 GTAV → CITYSCAPES

|          Methods          |        road        |       side.       |       buil.       | wall               |       fence       |        pole        |       light       |      sign      |       vege.       |       terr.       |      sky      |        per.        |       rider       |        car        |       truck       |        bus        |       train       |       moto.       |      bike      |        mIoU        |
| :------------------------: | :----------------: | :----------------: | :----------------: | ------------------ | :----------------: | :----------------: | :----------------: | :------------: | :----------------: | :----------------: | :------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :------------: | :----------------: |
|           Source           |        63.3        |        15.7        |        59.4        | 8.6                |        15.2        |        18.3        |        26.9        |      15.0      |        80.5        |        15.3        |      73.0      |        51.0        |        17.7        |        59.7        |        28.2        |        33.1        |        3.5        |        23.2        |      16.7      |        32.9        |
|            DSP            |        92.4        |        48.0        |        87.4        | 33.4               |        35.1        |        36.4        |        41.6        |      46.0      |        87.7        |        43.2        |      89.8      |        66.6        |        32.1        |        89.9        |        57.0        |        56.1        |        0.0        |        44.1        |      57.8      |        55.0        |
|            BDM            |        91.3        |        51.8        |        86.7        | 49.9               |        49.2        |        53.3        |        43.1        |      43.3      |        85.5        |        47.9        |      85.7      |        62.3        |        45.9        |        87.8        |        55.5        |        54.4        |        4.4        |        46.3        |      50.4      |        57.6        |
|            I2F            |        90.8        |        48.7        |        85.2        | 30.6               |        28.0        |        33.3        |        46.4        |      40.0      |        85.6        |        39.1        |      88.1      |        61.8        |        35.0        |        86.7        |        46.3        |        55.6        |        11.6        |        44.7        |      54.3      |        53.3        |
|            ADPL            |        93.4        |        60.6        |        87.5        | 45.3               |        32.6        |        37.3        |        43.3        |      55.5      |        87.2        |        44.8        |      88.0      |        64.5        |        34.2        |        88.3        |        52.6        |        61.8        |        49.8        |        41.8        |      59.4      |        59.4        |
|        Sepico(CNN)        |        95.2        |        67.8        |        88.7        | 41.4               |        38.4        |        43.4        |        55.5        |      63.2      |        88.6        |        46.4        |      88.3      |        73.1        |        49.0        |        91.4        |        63.2        |        60.4        |        0.0        |        45.2        |      60.0      |        61.0        |
|        Freedom(CNN)        |        90.9        |        54.1        |        87.8        | 44.1               |        32.6        |        45.2        |        51.4        |      57.1      |        88.6        |        42.6        |      89.5      |        68.8        |        40.0        |        89.7        |        58.4        |        62.6        |        55.3        |        47.7        |      40.0      |        61.3        |
| ***QuadMix(CNN)*** |      *97.1*      |      *78.0*      |      *90.4*      | *49.7*           |      *40.3*      |      *53.4*      |      *61.7*      |    *70.9*    |      *90.7*      |      *49.7*      |    *92.9*    |      *77.9*      |      *53.9*      |      *93.5*      |      *72.4*      |      *65.9*      |      *0.7*      |      *60.5*      |    *68.5*    |      *66.8*      |
|                            |                    |                    |                    |                    |                    |                    |                    |                |                    |                    |                |                    |                    |                    |                    |                    |                    |                    |                |                    |
|          DAFormer          |        95.7        |        70.2        |        89.4        | 53.5               |        48.1        |        49.6        |        55.8        |      59.4      |        89.9        |        47.9        |      92.5      |        72.2        |        44.7        |        92.3        |        74.5        |        78.2        |        65.1        |        55.9        |      61.8      |        68.3        |
|        Sepico(ViT)        |        96.7        |        76.7        |        89.7        | 55.5               |        49.5        |        53.2        |        60.0        |      64.5      |        90.2        |        50.3        |      90.8      |        74.5        |        44.2        |        93.3        |        77.0        |        79.5        |        63.6        |        61.0        |      65.3      |        70.3        |
|        Freedom(ViT)        |        96.7        |        74.8        |        90.9        | 58.1               |        49.0        |        57.5        |        63.4        | **71.4** |        91.6        |        52.1        | **94.4** |        78.4        |        53.1        |        94.1        |        83.9        |        85.2        |        72.5        |        62.8        | **68.9** |        73.6        |
| ***QuadMix(ViT)*** | ***97.5*** | ***80.9*** | ***91.6*** | ***62.3*** | ***57.6*** | ***58.2*** | ***64.5*** |    *71.2*    | ***91.7*** | ***52.3*** |    *94.3*    | ***80.0*** | ***55.9*** | ***94.6*** | ***86.3*** | ***90.5*** | ***82.3*** | ***65.1*** |    *68.1*    | ***76.1*** |

### 🔁 SYNTHIA → CITYSCAPES

|          Methods          |      road      |     side.     |     buil.     |     wall*     |     fence*     |       pole*       |     light     |        sign        |     vege.     |        sky        |        per.        |       rider       |      car      |        bus        |       motor       |      bike      |      mIoU(16)      |      mIoU(13)      |
| :------------------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :----------------: | :------------: | :----------------: | :------------: | :----------------: | :----------------: | :----------------: | :------------: | :----------------: | :----------------: | :------------: | :----------------: | :----------------: |
|           Source           |      36.3      |      14.6      |      68.8      |      9.2      |      0.2      |        24.4        |      5.6      |        9.7        |      69.0      |        79.4        |        52.5        |        11.3        |      49.8      |        9.5        |        11.0        |      20.7      |        33.7        |        29.5        |
|            DSP            |      86.4      |      42.0      |      82.0      |      2.1      |      1.8      |        34.0        |      31.6      |        33.2        |      87.2      |        88.5        |        64.1        |        31.9        |      83.8      |        64.4        |        28.8        |      54.0      |        51.0        |        59.9        |
|            BDM            |      91.0      |      55.8      |      86.9      |       —       |       —       |         —         |      58.3      |        44.7        |      85.8      |        85.7        |        84.1        |        40.3        |      86.0      |        55.2        |        45.0        |      50.6      |         —         |        66.8        |
|            I2F            |      84.9      |      44.7      |      82.2      |      9.1      |      1.9      |        36.2        |      42.1      |        40.2        |      83.8      |        84.2        |        68.9        |        35.3        |      83.0      |        49.8        |        30.1        |      52.4      |        51.8        |        60.1        |
|            ADPL            |      86.1      |      38.6      |      85.9      |      29.7      |      1.3      |        36.6        |      41.3      |        47.2        |      85.0      |        90.4        |        67.5        |        44.3        |      87.4      |        57.1        |        43.9        |      51.4      |        55.9        |        63.6        |
|        Sepico(CNN)        |      77.0      |      35.3      |      85.1      |      23.9      |      3.4      |        38.0        |      51.0      |        55.1        |      85.6      |        80.5        |        73.5        |        46.3        |      87.6      |        69.7        |        50.9        |      66.5      |        58.1        |        66.5        |
|        Freedom(CNN)        |      86.0      |      46.3      |      87.0      |      33.3      |      5.3      |        48.7        |      38.1      |        46.8        |      87.1      |        59.1        |        71.2        |        38.1        |      87.1      |        54.6        |        51.3        |      59.9      |        59.1        |        66.0        |
| ***QuadMix(CNN)*** |    *88.5*    |    *52.9*    |    *87.1*    |    *3.7*    |    *1.4*    |      *56.2*      |    *62.7*    |      *59.2*      |    *87.2*    |      *89.0*      |      *79.1*      |      *55.8*      |    *87.9*    |      *61.7*      |      *58.1*      |    *71.2*    |      *62.6*      |      *72.3*      |
|                            |                |                |                |                |                |                    |                |                    |                |                    |                    |                    |                |                    |                    |                |                    |                    |
|          DAFormer          |      84.5      |      40.7      |      88.4      |      41.5      |      6.5      |        50.0        |      55.0      |        54.6        |      86.0      |        89.8        |        73.2        |        48.2        |      87.2      |        53.2        |        53.9        |      61.7      |        60.9        |        67.4        |
|        Sepico(ViT)        |      87.0      | **52.6** |      88.5      |      40.6      | **10.6** |        49.8        |      57.0      |        55.4        |      56.8      |        86.2        |        75.4        |        52.7        | **92.4** |        78.9        |        53.0        |      62.6      |        64.3        |        71.4        |
|        Freedom(ViT)        | **89.4** |      50.8      | **89.3** | **48.8** |      9.3      |        57.3        | **65.1** |        60.1        | **89.9** |        93.7        |        79.4        |        51.6        |      90.5      |        66.0        |        62.3        | **68.1** |        67.0        |        73.6        |
| ***QuadMix(ViT)*** |    *88.1*    |    *51.2*    |    *88.9*    |    *46.7*    |    *7.9*    | ***58.6*** |    *64.7*    | ***63.7*** |    *88.1*    | ***93.9*** | ***81.3*** | ***56.6*** |    *90.3*    | ***66.9*** | ***66.8*** |    *66.0*    | ***67.5*** | ***74.3*** |

## ⚙️ Environment Setup

##### **A unified environment applicable to both image and video UDASS.**

1. create conda environment

```bash
conda create -n UDAVSS_py38 python=3.8 -y
conda activate UDAVSS_py38
conda install -c menpo opencv
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

2. clone the [ADVENT repo](https://leftgithub.com/valeoai/ADVENT) （only required for video UDASS）

```bash
git clone https://github.com/valeoai/ADVENT
pip install -e ./VIDEO/ADVENT
```

3. install requirements

```bash
pip install -r ./requirements.txt
```

4. resample2d dependency:（only required for video UDASS）

```
cd ./video_udass/VIDEO/tps/utils/resample2d_package
python setup.py build
python setup.py install
```

5. install mmcv-full, this command compiles mmcv locally and may take some time

```shell
pip install mmcv-full==1.3.7  # requires other packeges to be installed first
```

6. Please download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training) from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## 📁 Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to `/image_udass/seg/data/cityscapes`.

**GTA:** Please, download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `/image_udass/seg/data/gta`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from [here](http://synthia-dataset.net/downloads/) and extract it to `/image_udass/seg/data/synthia`.

The final folder structure should look like this:

```none
seg
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the train IDs and to generate the class index for RCS:

```shell
cd ./Unified_UDASS/udass/image_udass/seg
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## ✅ Evaluation

First, pls `cd ./Unified_UDASS/udass/image_udass/seg`

A trained model can be evaluated using:

|               Task               |                                                                                      Command Usage in `test.sh`                                                                                      |                 Modification in `__init__.py`                 |
| :-------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------: |
|   transformer_GTA → Cityscapes   |   ``bash<br>TEST_ROOT=$1<br>CONFIG_FILE="${TEST_ROOT}/gtaHR2csHR_udass_hrda_650a8.py"<br>CHECKPOINT_FILE="${TEST_ROOT}/udass_image_gta_trans.pth"<br>SHOW_DIR="${TEST_ROOT}/preds_gta_trans_udass"``   |   ``python<br>from mmseg.models.uda.ok.dacs_gta import DACS``   |
| transformer_Synthia → Cityscapes | ``bash<br>TEST_ROOT=$1<br>CONFIG_FILE="${TEST_ROOT}/synthiaHR2csHR_udass_hrda_650a8.py"<br>CHECKPOINT_FILE="${TEST_ROOT}/udass_image_syn_trans.pth"<br>SHOW_DIR="${TEST_ROOT}/preds_syn_trans_udass"`` |   ``python<br>from mmseg.models.uda.ok.dacs_syn import DACS``   |
|     cnn_Synthia → Cityscapes     |    ``bash<br>TEST_ROOT=$1<br>CONFIG_FILE="${TEST_ROOT}/synthiaHR2csHR_udass_hrda_cnn.py"<br>CHECKPOINT_FILE="${TEST_ROOT}/udass_image_syn_cnn.pth"<br>SHOW_DIR="${TEST_ROOT}/preds_syn_cnn_udass"``    | ``python<br>from mmseg.models.uda.ok.dacs_syn_cnn import DACS`` |
|       cnn_GTA → Cityscapes       |      ``bash<br>TEST_ROOT=$1<br>CONFIG_FILE="${TEST_ROOT}/gtaHR2csHR_udass_hrda_cnn.py"<br>CHECKPOINT_FILE="${TEST_ROOT}/udass_image_gta_cnn.pth"<br>SHOW_DIR="${TEST_ROOT}/preds_gta_cnn_udass"``      | ``python<br>from mmseg.models.uda.ok.dacs_gta_cnn import DACS`` |

The checkpoints should be downloaded and be put in ./work_dirs.

Below, we provide checkpoints of UDASS(HRDA) for the different benchmarks.

|               Checkpoint Name               | Download Link                                                                           |
| :-----------------------------------------: | --------------------------------------------------------------------------------------- |
|   image-UDASS (ViT) for GTA → Cityscapes   | [Link](https://drive.google.com/file/d/1VYlG0f92Y8VAv712-i5f4GEIwTqbURqf/view?usp=sharing) |
| image-UDASS (ViT) for Synthia → Cityscapes | [Link](https://drive.google.com/file/d/1ll6BAqoexkNDOpLJ3eaJRIBgkysnMwsZ/view?usp=sharing) |
|   image-UDASS (CNN) for GTA → Cityscapes   | [Link](https://drive.google.com/file/d/15ryaQVPAcwuvx42N4ag-ilUyHizRpB5s/view?usp=sharing) |
| image-UDASS (CNN) for Synthia → Cityscapes | [Link](https://drive.google.com/file/d/1JgStA157b85Ueh7S_WJUP6iMHhUkLybW/view?usp=sharing) |

The predictions are saved for inspection to  `work_dirs/preds` and the mIoU of the model is printed to the console.

When training a model on Synthia→Cityscapes, please note that the evaluation script calculates the mIoU for all 19 Cityscapes classes.

However, Synthia contains only labels for 16 of these classes. Therefore, it is a [common ](https://github.com/lhoyer/MIC/tree/master/seg)practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16 classes. As the Iou for the 3 missing classes is 0, you can do the conversion  `mIoU16 = mIoU19 * 19 / 16`.

The predictions can be submitted to the public evaluation server of the respective dataset to obtain the test score.

##### The evalution code can be run in parallel with Train codes.

## ✅ Training

For the experiments in our paper, we use a script to automatically generate and train the configs.

Specifically, pls first  `cd ./Unified_UDASS/udass/image_udass/seg`,  then:

| Task            | Command                                                                       | Modification in `train.py`                                                                                                                                                                     | Modification in `__init__.py`                       | Modification in `dacs_xxx.py` |
| --------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------- |
| transformer_gta | `python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py`     | `cfg.resume_from = './work_dirs/local-basic/240810_0952_gtaHR2csHR_mic_hrda_s2_a891a/iter_4000.pth'`                                                                                           | `from mmseg.models.uda.ok.dacs_gta import DACS`     | `self.local_iter = 4000`      |
| transformer_syn | `python run_experiments.py --config configs/mic/synthiaHR2csHR_mic_hrda.py` | `cfg.resume_from = './work_dirs/local-basic/240810_0955_synthiaHR2csHR_mic_hrda_s2_ade8e/iter_3000.pth'`                                                                                       | `from mmseg.models.uda.ok.dacs_syn import DACS`     | `self.local_iter = 3000`      |
| cnn_syn         | `python run_experiments.py --exp 821`                                       | `cfg.resume_from = './work_dirs/local-basic/240810_1333_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_b6485/iter_6000.pth'` | `from mmseg.models.uda.ok.dacs_syn_cnn import DACS` | `self.local_iter = 6000`      |
| cnn_gta         | `python run_experiments.py --exp 811`                                       | `cfg.resume_from = './work_dirs/local-basic/240810_1332_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_710e9/iter_6000.pth'` | `from mmseg.models.uda.ok.dacs_gta_cnn import DACS` | `self.local_iter = 6000`      |

More information about the available experiments and their assigned IDs, can be found in [experiments.py](experiments.py). The generated configs will be stored in `configs/generated/`.

## 💾 Pretrained checkpoints for training

Our training follows the same paradigm as [MIC ](https://github.com/lhoyer/MIC/tree/master/seg)in the first *n* iterations, allowing direct loading of MIC's intermediate checkpoints to accelerate training (in this project).

Below, we first provide MIC's intermediate checkpoints for the different benchmarks.

Download [checkpoints folder](https://pan.baidu.com/s/1nB0Ii3bxlyd9adreiRi_HQ?pwd=hphf) (code: hphf) for  transformer_gta,  transformer_syn,  cnn_syn,  cnn_gta, and placed it in `./work_dirs/local-basic`.

Alternatively, training can aslo start from iteration 0.

## Acknowledgements

Image-UDASS is based on the following open-source projects. We thank their authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC/tree/master)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)

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
