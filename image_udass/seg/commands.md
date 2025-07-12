cd /home/sysmanager/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/udass/image_udass/seg




transformer_gta
python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py
cfg.resume_from = './work_dirs/local-basic/240810_0952_gtaHR2csHR_mic_hrda_s2_a891a/iter_4000.pth'
self.local_iter = 4000




transformer_syn
python run_experiments.py --config configs/mic/synthiaHR2csHR_mic_hrda.py
cfg.resume_from = './work_dirs/local-basic/240810_0955_synthiaHR2csHR_mic_hrda_s2_ade8e/iter_3000.pth'
self.local_iter = 3000




cnn_syn
python run_experiments.py --exp 821
cfg.resume_from = './work_dirs/local-exp821/240810_1333_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_b6485/iter_6000.pth'
self.local_iter = 6000





cnn_gta
python run_experiments.py --exp 811
cfg.resume_from = './work_dirs/local-exp811/240810_1332_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_710e9/iter_6000.pth'
self.local_iter = 6000



