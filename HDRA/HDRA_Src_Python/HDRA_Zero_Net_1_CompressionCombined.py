# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 10.11.2020
# =========================================

import pandas as pd
import numpy as np
from natsort import natsorted
import glob


# read HDRA_Family ID
HDRA_FamilyID = pd.read_csv('../HDRA_Data/HDRA_Raw/HDRA_Gen_FamilyID.csv', low_memory=False, header=None)
HDRA_FamilyID = HDRA_FamilyID[0].astype(str).tolist()
print(HDRA_FamilyID)
# load Compressed Data
Net_1_Compressed_fileList = natsorted(glob.glob("../HDRA_Data/HDRA_ModelMetaData/"
                            "HDRA_Net_1/HDRA_Net_1_EncData/"+"*.csv"))
# Net_1_Compressed_fileList = Net_1_Compressed_fileList[0:5]
Net_1_Compressed_csv = pd.concat([pd.read_csv(f, header=None) for f in Net_1_Compressed_fileList], axis=1)
Net_1_Compressed_csv = Net_1_Compressed_csv.transpose()
# print(Net_1_Compressed_csv)

# Concat HDRA_Family ID with compressed data
# Net_1_Compressed_csv.columns = HDRA_FamilyID
# print(Net_1_Compressed_csv)

Net_1_Compressed_csv.to_csv("../HDRA_Data/HDRA_CompressedData/HDRA_Compress_1.csv",
                       index=False, header=HDRA_FamilyID)




