# UDASS

The official repository for **Unified Domain Adaptive Semantic Segmentation (UDASS)**.  
[[📄 Paper (arXiv)](https://arxiv.org/abs/2311.13254)]  
[[🎬 Demo Video](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing)]

## 🗞️ News

🎉 Our paper has been **accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2025.  
Please stay tuned for the journal version and updated DOI!

---

## 📝 Overview

**Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS)** aims to transfer supervision from a labeled source domain to an unlabeled target domain. Most existing UDA-SS works focus on images, while recent studies have extended this to video by modeling the temporal dimension. Despite sharing the core challenge, i.e. overcoming domain distribution shift, the two research lines have largely developed in isolation.

**This separation introduces several limitations:**

1. Insights remain fragmented, lacking a unified understanding of the problem and potential solutions.  
2. Unified methods, techniques, and best practices cannot be established, causing redundant efforts and missed opportunities.  
3. Advances in one domain (image or video) cannot effectively transfer to the other, leading to suboptimal performance.

**Our motivation:** We advocate for unifying the study of UDA-SS across image and video settings, enabling comprehensive understanding, synergistic advances, and efficient knowledge sharing.

To this end, we introduce a general data augmentation perspective as a unifying conceptual framework. Specifically, we propose **Quad-directional Mixup (QuadMix)**, which performs intra- and inter-domain mixing in feature space through four directional paths. To address temporal shifts in video, we incorporate **optical flow-guided spatio-temporal aggregation** for fine-grained domain alignment.

**Extensive experiments** on four challenging UDA-SS benchmarks show that our method outperforms state-of-the-art approaches by a large margin.

*Keywords: Unified domain adaptation, semantic segmentation, QuadMix, flow-guided spatio-temporal aggregation.*

---

## 🎥 Click to Watch More Qualitative Results

[![Watch demo video](https://github.com/ZHE-SAPI/UDASS/blob/main/Unified-UDASS.jpg?raw=true)](https://youtu.be/DgrZYkebhs0)

You can also find the demo video on:  
- [Bilibili 视频](https://www.bilibili.com/video/BV1ZgtMejErB/?vd_source=ae767173839d1c3a41173ad40cc34d53)  
- [Google Drive](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing)

> 💡 Please select **HD (1080p)** for clearer visualizations.

---

## 🧩 UDASS for Image Scenarios

Source code for image-based UDA-SS is located in the [`/image_udass`](https://github.com/ZHE-SAPI/UDASS/tree/main/image_udass) directory.

For setup and training instructions, refer to the [seg/README.md](https://github.com/ZHE-SAPI/UDASS/blob/main/image_udass/seg/README.md).

---

## 🎞️ UDASS for Video Scenarios

Source code for video-based UDA-SS is located in the [`/video_udass`](https://github.com/ZHE-SAPI/UDASS/tree/main/video_udass) directory.

For setup and training instructions, refer to the [VIDEO/README.md](https://github.com/ZHE-SAPI/UDASS/blob/main/video_udass/VIDEO/README.md).

---

## 📚 Citation

If you find UDASS useful in your research, please consider citing:

```bibtex
@misc{zhang2024udass,
      title={Unified Domain Adaptive Semantic Segmentation}, 
      author={Zhe Zhang and Gaochang Wu and Jing Zhang and Chunhua Shen and Dacheng Tao and Tianyou Chai},
      year={2024},
      eprint={2311.13254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
