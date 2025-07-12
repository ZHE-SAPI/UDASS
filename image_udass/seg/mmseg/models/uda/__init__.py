# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MinEnt and AdvSeg
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.advseg import AdvSeg
from mmseg.models.uda.minent import MinEnt

from mmseg.models.uda.ok.dacs_gta import DACS # transformer_gta
# from mmseg.models.uda.ok.dacs_syn import DACS # transformer_syn
# from mmseg.models.uda.ok.dacs_syn_cnn import DACS # cnn_syn
# from mmseg.models.uda.ok.dacs_gta_cnn import DACS # cnn_gta


__all__ = ['DACS', 'MinEnt', 'AdvSeg']
