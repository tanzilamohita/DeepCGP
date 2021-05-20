# =========================================
# Tanzila Islam
# PhD Student, Iwate University
# Email: tanzilamohita@gmail.com
# Created Date: 20.06.2020
# =========================================

import pandas as pd
import numpy as np

# read phenotype data
height_data = pd.read_csv('../C7AIR_Data/C7AIR_Raw/C7AIR_Phenotype.csv', low_memory=False)
height_data = height_data.transpose()
height_data = height_data.iloc[0:2, :]   # 1st one column with id
trait_id = height_data.iloc[0]
phen_data = height_data[1:]
phen_data.columns = trait_id
phen_data = phen_data.dropna(axis=1)  # remove missing entries
print(phen_data)
y = np.array(phen_data)
print(y)

# read Net_1 combined compressed data
Net_1_compressed_data = pd.read_csv('../C7AIR_Data/C7AIR_CompressedData/'
                                    'C7AIR_Net_1_CompressedData.csv',
                                    low_memory=False, header=None)
Net_1_compressed_data = Net_1_compressed_data.transpose()
Net_1_compressed_data.columns = trait_id
print(Net_1_compressed_data)
Net_1_compressed_data.to_csv("../C7AIR_Data/C7AIR_CompressedData/"
                             "C7AIR_Compress_1_Pheno.csv", index=False)

# read Net_2 combined compressed data
Net_2_compressed_data = pd.read_csv('../C7AIR_Data/C7AIR_CompressedData/'
                                    'C7AIR_Net_2_CompressedData.csv',
                                    low_memory=False, header=None)
Net_2_compressed_data = Net_2_compressed_data.transpose()
Net_2_compressed_data.columns = trait_id
print(Net_2_compressed_data)
Net_2_compressed_data.to_csv("../C7AIR_Data/C7AIR_CompressedData/"
                             "C7AIR_Compress_2_Pheno.csv", index=False)







