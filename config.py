import os.path as osp
import numpy as np
from easydict import EasyDict
from tps.utils import project_root, project_root_ADVENT
from ADVENT.advent.utils.serialization import yaml_load

cfg = EasyDict()


cfg.N_SEGMENTS = 4
cfg.tar_loss_weight = 0.6

cfg.tensorboard = True
cfg.generate_tcsm_synthia_path = ''
cfg.prototype_path = str(project_root / 'experiments_result/prototype')
cfg.train_thred = 0.0
cfg.proto_temperature = 1.0
cfg.regular_type = 'MRKLD' # 'MRENT|MRKLD'
cfg.regular_w = 0.1
cfg.ema_w = 0.5

cfg.lamda = 1.0
cfg.loss_cts_W = 0.5
cfg.proto_consistW = 10.0
cfg.pixel_weight = "threshold_uniform"
cfg.dsp_hard_loss = True # False || True

cfg.NUM_WORKERS = 4
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments_result'
cfg.project = project_root
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')

cfg.TENSORBOARD_DIR = ''
cfg.proto_resume = False
cfg.alpha_mix = 0.5
cfg.mix_lay2_out = True # feature fusion
cfg.mix_lay3_out = True # feature fusion
cfg.mix_lay6_out = True # feature fusion

cfg.generate_threshold_path = ''
cfg.ignore_index = 250

cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 1

cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
# cfg.TRAIN.INPUT_SIZE_SOURCE = (1024, 512)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.MULTI_LEVEL = True


cfg.TRAIN.STAGE1 = False 
cfg.TRAIN.STAGE2 = False 
cfg.TRAIN.STAGE3 = False #2fff 3fff 4fff; 5tff 6tff 7tff 8ftf 9fft


cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.SAVE_PRED_EVERY = 100
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_VIZRATE = 100
cfg.TRAIN.LAMBDA_T = 1.0

# TPS
cfg.TRAIN.DA_METHOD = 'SourceOnly'
cfg.NUM_CLASSES = 15
cfg.SOURCE = 'Viper'
cfg.TARGET = 'CityscapesSeq'
cfg.TRAIN.INFO_TARGET = str(project_root / 'tps/dataset/CityscapesSeq_list/info_Viper.json')
cfg.DATA_LIST_SOURCE = str(project_root / 'tps/dataset/Viper_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'tps/dataset/CityscapesSeq_list/{}.txt')
cfg.DATA_DIRECTORY_SOURCE = ''
cfg.DATA_DIRECTORY_TARGET = '../../video_seg/TPS/data/Cityscapes'
cfg.TRAIN.MODEL = 'ACCEL_DeepLabv2'
cfg.TRAIN.flow_path_src = ''
cfg.TRAIN.flow_path = '../../video_seg/TPS/tps/data/estimated_optical_flow_cityscapes_seq_train'
cfg.TRAIN.INTERVAL = 1
cfg.TRAIN.THRESHOLD = 0.0
cfg.TRAIN.SCALING_RATIO = (0.8, 1.2)
cfg.TRAIN.RESTORE_FROM = 'pretrained_models/DeepLab_resnet_pretrained_imagenet.pth'
cfg.TRAIN.RESTORE_FROM_viper = 'pretrained_models/tps_viper2city_pretrained.pth'
cfg.TRAIN.RESTORE_FROM_synthia = 'pretrained_models/tps_syn2city_pretrained.pth'
cfg.TRAIN.RESTORE_FROM_5539 = 'pretrained_models/synthia_14400_cd_55.39_model_14400.pth'

cfg.TRAIN.SNAPSHOT_DIR_DSP = ''
cfg.TRAIN.SNAPSHOT_DIR_DSP_CD = ''
cfg.TRAIN.SNAPSHOT_DIR_PRODA = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_1 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_2 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_3 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_4 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_5 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_6 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_7 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_8 = ''
cfg.TRAIN.SNAPSHOT_DIR_ACC_9 = ''
  
cfg.TRAIN.SNAPSHOT_DIR_TCSM_1 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_2 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_3 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_4 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_5 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_6 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_7 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_8 = ''
cfg.TRAIN.SNAPSHOT_DIR_TCSM_9 = ''

cfg.TRAIN.RESTORE_FROM_ACC0 = ''
cfg.TRAIN.RESTORE_FROM_ACC1 = ''
cfg.TRAIN.RESTORE_FROM_ACC2 = ''
cfg.TRAIN.RESTORE_FROM_ACC3 = ''

cfg.TRAIN.RESTORE_FROM_TCSM0 = ''
cfg.TRAIN.RESTORE_FROM_TCSM1 = ''
cfg.TRAIN.RESTORE_FROM_TCSM2 = ''
cfg.TRAIN.RESTORE_FROM_TCSM3 = ''

cfg.TRAIN.TENSORBOARD_LOGDIR = '' # D
cfg.TRAIN.TENSORBOARD_LOGDIR_CD = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_PRODA = ''

cfg.TRAIN.TENSORBOARD_LOGDIR1  = ''
cfg.TRAIN.TENSORBOARD_LOGDIR2 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR3 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR4 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR5 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR6 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR7 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR8 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR9 = ''

cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM1 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM2 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM3 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM4 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM5 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM6 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM7 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM8 = ''
cfg.TRAIN.TENSORBOARD_LOGDIR_TCSM9 = ''





# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'video_single'  # {'video_single', 'video_best', 'video_best_tcsm'}
cfg.TEST.MODEL = ('ACCEL_DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_TCSM_8 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_TCSM_8/model_tcsm_26100.pth',)  # '../../experiments/snapshots/tps_syn2city/model_26100.pth'
cfg.TEST.SNAPSHOT_TCSM_1 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_TCSM_1/model_tcsm_26100.pth',)  # '../../experiments/snapshots/tps_syn2city/model_26100.pth'
cfg.TEST.SNAPSHOT_TCSM_7 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_TCSM_7/model_tcsm_26100.pth',)  # '../../experiments/snapshots/tps_syn2city/model_26100.pth'
cfg.TEST.RESTORE_FROM_SYN1 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/pretrained_models/tps_syn2city_pretrained.pth',)  # '../../experiments/snapshots/tps_syn2city/model_26100.pth'
cfg.TEST.RESTORE_FROM_SYN2 = ('../../experiments/snapshots/tps_syn2city/model_26100.pth',)
cfg.TEST.RESTORE_FROM_SYN2_single = ('../../experiments/snapshots/tps_syn2city',)


cfg.TEST.SNAPSHOT_DIR_DSP_CD_synthina_nolongtail = ('./experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_CD_nolongtail',)
cfg.TEST.SNAPSHOT_DIR_DSP_CD_synthina_nofusion = ('./experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_CD_nofusion',)

cfg.TEST.RESTORE_FROM_SYN = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/tps_syn2city_SourceOnly/model_29000.pth',)  # '../../experiments/snapshots/tps_syn2city/model_26100.pth'
cfg.TEST.RESTORE_FROM_prototype = ('',)

cfg.TEST.RESTORE_FROM_SYN_TCSM = ''

# cfg.TEST.SNAPSHOT_DIR = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_ACC_1',)  # used in 'best' mode
# cfg.TEST.SNAPSHOT_DIR = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_ACC_7',)  # used in 'best' mode
# cfg.TEST.SNAPSHOT_DIR = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_seg/TPS/experiments_result/snapshots/tps_syn2city_ACC_8',)  # used in 'best' mode
# cfg.TEST.SNAPSHOT_DIR = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/syn2city_TPS',)  # used in 'best' mode
# cfg.TEST.SNAPSHOT_DIR is cfg.TEST.SNAPSHOT_DIR_C
cfg.TEST.SNAPSHOT_DIR_DSP_CD = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_CD',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_DIR_PRODA = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/tps_syn2city_PRODA',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_DIR_PRODA_consis = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_PRODA_Pseudo',)  # used in 'best' mode 
cfg.TEST.SNAPSHOT_DIR_PRODA_pesudo = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_PRODA__PESUDO',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_DIR_PRODA_pesudo_cd = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_PRODA__PESUDO_cd',)  # used in 'best' mode

cfg.TEST.SNAPSHOT_DIR_PRODA_pesudo_cd_02_pt = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_PRODA__PESUDO_cd_02_5539_pt',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_DIR_PRODA_pesudo_cd_02 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_PRODA__PESUDO_cd_02_5539',)  # used in 'best' mode

cfg.TEST.SNAPSHOT_DIR_C_single = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/syn2city_TPS/model_16000.pth',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_DIR_DSP_CD_single = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_CD/model_14400.pth',)  # used in 'best' mode
                                        

cfg.TEST.SNAPSHOT_DIR_DSP_CD_syn_TPSparam = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_CD_TPSparam',)





cfg.TEST.DAVSS_DSF_cd = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_noDFF = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_noDFF',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_nodual = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_nodual',)


cfg.TEST.DAVSS_DSF_cd_EMA_longtail3 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_EMA_longtail3',)
cfg.TEST.DAVSS_DSF_cd_EMA_woLongtail4 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_EMA_woLongtail4',)
cfg.TEST.DAVSS_DSF_DFF_cd_EMA_longtail6 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_EMA_longtail6_add',)
cfg.TEST.DAVSS_DSF_DFF_cd_EMA_longtail65 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_EMA_longtail6_conv',)
cfg.TEST.DAVSS_DSF_DFF_cd_EMA_longtail66 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_EMA_longtail66_deepfusion_add',)


cfg.TEST.DAVSS_DSF_cd_longtail3 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_longtail3',)
cfg.TEST.DAVSS_DSF_cd_woLongtail4 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_woLongtail4',)
cfg.TEST.DAVSS_DSF_DFF_cd_longtail6 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_longtail6_add',)
cfg.TEST.DAVSS_DSF_DFF_cd_longtail65 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_longtail6_conv',)
cfg.TEST.DAVSS_DSF_DFF_cd_longtail66 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_DFF_cd_longtail66_deepfusion_add',)

cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF',)
cfg.TEST.SNAPSHOT_DIR = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_norescale',)

cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_0005 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_0005',)

cfg.TEST.TPS_norescale_tagg_12_pro_mmd01_1 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_norescale_tagg_12_pro_mmd01_1',)

cfg.TEST.TPS_norescale_tagg_12_pro_mmd1_2 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_norescale_tagg_12_pro_mmd1_2',)


cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg1 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg_mmd0_1',)

cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_syn2city_noDFF_freeze_tagg = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze_tagg',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_syn2city_noDFF_freeze_tagg1 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze_tagg_mmd0_1',)

cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg_mmd1_lt1_samhalf = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg_mmd1_lt1_samhalf',)
cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze_tagg_samhalf = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_synthia_noDFF_freeze_tagg_samhalf',)

cfg.TEST.DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg_cmom = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_flowdbnoflip_PARM_viper_noDFF_freeze_tagg_cmom',)



# ablation
cfg.TEST.DAVSS_DSF_cd_ablation_5 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_5',)
cfg.TEST.DAVSS_DSF_cd_ablation_13 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_13',)

cfg.TEST.DAVSS_DSF_cd_ablation_7 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_7',)
cfg.TEST.DAVSS_DSF_cd_ablation_10 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_10',)
cfg.TEST.DAVSS_DSF_cd_ablation_17 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_17',)
cfg.TEST.DAVSS_DSF_cd_ablation_175 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_175',)

cfg.TEST.DAVSS_DSF_cd_ablation_22 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/Viper2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_22',)
cfg.TEST.DAVSS_DSF_cd_ablation_23 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_23',)
cfg.TEST.DAVSS_DSF_cd_ablation_24 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_24',)
cfg.TEST.DAVSS_DSF_cd_ablation_25 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_25',)
cfg.TEST.DAVSS_DSF_cd_ablation_26 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_26',)
cfg.TEST.DAVSS_DSF_cd_ablation_27 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_27',)
cfg.TEST.DAVSS_DSF_cd_ablation_28 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_28',)
cfg.TEST.DAVSS_DSF_cd_ablation_29 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_29',)
cfg.TEST.DAVSS_DSF_cd_ablation_36 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_36',)

cfg.TEST.DAVSS_DSF_cd_ablation_43 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_43',)
cfg.TEST.DAVSS_DSF_cd_ablation_42 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_42',)
cfg.TEST.DAVSS_DSF_cd_ablation_40 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_40',)
cfg.TEST.DAVSS_DSF_cd_ablation_48 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_48',)
cfg.TEST.DAVSS_DSF_cd_ablation_49 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DAVSS_DSF_cd_ablation_49',)

#   ablation

cfg.TEST.DAVSS_DSF_d_2 = ('/home/ZZF/video_tps/TPS/experiments_result/snapshots/SynthiaSeq2CityscapesSeq_ACCEL_DeepLabv2_TPS_DSP_D_2',)

cfg.TEST.SNAPSHOT_STEP = 200  # used in 'best' mode
cfg.TEST.SNAPSHOT_START_ITER = 200  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 40000  # used in 'best' mode
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'tps/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True
cfg.TEST.INTERVAL = 1
cfg.TEST.flow_path = '../../video_seg/TPS/tps/data/estimated_optical_flow_cityscapes_seq_val'

cfg.TEST.SNAPSHOT_DIR_ACC_1 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_2 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_3 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_4 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_5 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_6 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_7 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_8 = ('',)
cfg.TEST.SNAPSHOT_DIR_ACC_9 = ('',)

cfg.TEST.SNAPSHOT_DIR_TCSM_1 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_2 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_3 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_4 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_5 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_6 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_7 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_8 = ('',)
cfg.TEST.SNAPSHOT_DIR_TCSM_9 = ('',)



cfg.TEST.RESTORE_FROM_ACC0 = ('',)
cfg.TEST.RESTORE_FROM_ACC1 = ('',)
cfg.TEST.RESTORE_FROM_ACC2 = ('',)
cfg.TEST.RESTORE_FROM_ACC3 = ('',)

cfg.TEST.RESTORE_FROM_TCSM0 = ('',)
cfg.TEST.RESTORE_FROM_TCSM1 = ('',)
cfg.TEST.RESTORE_FROM_TCSM2 = ('',)
cfg.TEST.RESTORE_FROM_TCSM3 = ('',)

# calculate prototype
cfg.proto_momentum = 0.1 # 0.9
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
