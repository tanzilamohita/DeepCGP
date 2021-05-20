# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 21.10.2020
# =========================================

import glob
import numpy as np
import pandas as pd
from natsort import natsorted

# load Net_1 Compressed Data
Net_1_Compressed_fileList = natsorted(glob.glob("../C7AIR_Data/C7AIR_ModelMetaData/"
                                                "C7AIR_Net_1/C7AIR_Net_1_EncData/"
                                                +"*.csv"))
Net_1_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           for f in Net_1_Compressed_fileList], axis=1)
print(Net_1_Compressed_combined_csv)
# export to csv
Net_1_Compressed_combined_csv.to_csv("../C7AIR_Data/C7AIR_CompressedData/"
                                     "C7AIR_Net_1_CompressedData.csv", index=False,
                                     header=None, float_format='%1.0f')

# load Net_2 Compressed Data
Net_2_Compressed_fileList = natsorted(glob.glob("../C7AIR_Data/C7AIR_ModelMetaData/"
                                                "C7AIR_Net_2/C7AIR_Net_2_EncData/"
                                                +"*.csv"))
Net_2_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           for f in Net_2_Compressed_fileList], axis=1)
print(Net_2_Compressed_combined_csv)
# export to csv
Net_2_Compressed_combined_csv.to_csv("../C7AIR_Data/C7AIR_CompressedData/"
                                     "C7AIR_Net_2_CompressedData.csv", index=False,
                                     header=None, float_format='%1.0f')





