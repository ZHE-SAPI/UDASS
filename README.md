# UDASS

The official repository for **Unified Domain Adaptive Semantic Segmentation (TPAMI 2025)**.

|    Resource     |                                                             Link                                                             |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------: |
|      📄 PDF      | [Paper and Supplementary Material](https://github.com/ZHE-SAPI/UDASS/blob/master/Paper%20and%20Supplementary%20Material.pdf) |
|  📄 IEEE Xplore  |                            [IEEE Xplore: 10972076](https://ieeexplore.ieee.org/document/10972076)                            |
| 📄 Paper (Arxiv) |                                          [Arxiv](https://arxiv.org/abs/2311.13254)                                           |
|  🎬 Video Demo   |       [Demo Video (Google Drive)](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing)        |


## 🧩 UDASS for Image Scenarios

Source code for image-based UDA-SS is located in the [`/image_udass`](https://github.com/ZHE-SAPI/UDASS/tree/master/image_udass) directory.

For setup and training instructions, refer to the [image_udass/README.md](https://github.com/ZHE-SAPI/UDASS/tree/master/image_udass/seg/README.md).

---

## 🧩UDASS for Video Scenarios

Source code for video-based UDA-SS is located in the [`/video_udass`](https://github.com/ZHE-SAPI/UDASS/tree/master/video_udass) directory.

For setup and training instructions, refer to the [video_udass/README.md](https://github.com/ZHE-SAPI/UDASS/tree/master/video_udass/VIDEO/README.md).

---


## 🗞️ News

- 🎉 **2025.07.12**: We have released the complete source code! Feel free to contact us if you have any questions, we are happy to discuss!
- 🎉 If you find **UDASS** helpful in your research, please consider giving us a ⭐ on GitHub and citing our work in your publications!
- 🎉 **2025.04.25**: Our paper has been **accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2025!

## 📝 Abstract

**Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS)** aims to transfer supervision from a labeled source domain to an unlabeled target domain. Most existing UDA-SS works focus on images, while recent studies have extended this to video by modeling the temporal dimension. Despite sharing the core challenge, i.e. overcoming domain distribution shift, the two research lines have largely developed in isolation. **This separation introduces several limitations:**

1. Insights remain fragmented, lacking a unified understanding of the problem and potential solutions.
2. Unified methods, techniques, and best practices cannot be established, causing redundant efforts and missed opportunities.
3. Advances in one domain (image or video) cannot effectively transfer to the other, leading to suboptimal performance.

**Our motivation:** We advocate for unifying the study of UDA-SS across image and video settings, enabling comprehensive understanding, synergistic advances, and efficient knowledge sharing.

To this end, we introduce a general data augmentation perspective as a unifying conceptual framework. Specifically, we propose **Quad-directional Mixup (QuadMix)**, which performs intra- and inter-domain mixing in feature space through four directional paths. To address temporal shifts in video, we incorporate **optical flow-guided spatio-temporal aggregation** for fine-grained domain alignment.

**Extensive experiments** on four challenging UDA-SS benchmarks show that our method outperforms state-of-the-art approaches by a large margin.

*Keywords: Unified domain adaptation, semantic segmentation, QuadMix, flow-guided spatio-temporal aggregation.*

---

## 🧩Click to Watch More Qualitative Results

[![Watch demo video](https://github.com/ZHE-SAPI/UDASS/blob/master/Unified-UDASS.jpg)](https://youtu.be/DgrZYkebhs0)

You can also find the demo video on:

- [Bilibili 视频](https://www.bilibili.com/video/BV1ZgtMejErB/?vd_source=ae767173839d1c3a41173ad40cc34d53)
- [Google Drive](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing)

> 💡 Please select **HD (1080p)** for clearer visualizations.

---


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
