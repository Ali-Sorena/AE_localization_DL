import os
import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt

# Package of addresses
hForce = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/Data/Resource - Coding (CSV)/ae_pipe/1 High Force/High Force - waveform/CSV Files/"
mForce = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/Data/Resource - Coding (CSV)/ae_pipe/2 Medium Force/Medium Force - waveform/CSV Files/"
lForce = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/Data/Resource - Coding (CSV)/ae_pipe/3 Low Force/Low Force - waveform/CSV Files/"

sourceloc = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/CSV_XL/Locations.csv"
sourcehdr = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/CSV_XL/Headers2.csv"

results = "C:/Users/So.re.na/Desktop/Fraunhofer IEG/Data/Sensors CSV/Result data/"

# Choosing destinations
# workdir = lForce  # Casewise choice
SensorsIndex = np.array([[1, 2, 3, 4]])   #adding Sensor labeling
# print(SensorsIndex.shape)
SensorsH, SensorsM, SensorsL, AllData = ([] for k in range(4))  # Main body
locdata = loadtxt(sourceloc, delimiter=",")  # locations
locdata = locdata.transpose()
locdatanp = np.array(locdata)

###### Part H ####################################################################################################################

data_Sensor1, data_Sensor2, data_Sensor3, data_Sensor4, data = ([] for i in range(5))  # Transient data
Sensor1, Sensor2, Sensor3, Sensor4 = ([] for j in range(4))  # Main body

for filename in os.listdir(hForce):  # Main body
    csv_file_path = os.path.join(hForce, filename)
    data = loadtxt(csv_file_path, delimiter=",")
    data = np.concatenate((data, SensorsIndex))
    data = data.transpose()
    data_Sensor1.append(data[0]); data_Sensor2.append(data[1]); data_Sensor3.append(data[2]); data_Sensor4.append(data[3])


# adding locations
data_Sensor1 = np.transpose(data_Sensor1); data_Sensor2 = np.transpose(data_Sensor2)
data_Sensor3 = np.transpose(data_Sensor3); data_Sensor4 = np.transpose(data_Sensor4)

Sensor1 = np.concatenate((data_Sensor1, locdatanp)); Sensor1 = np.transpose(Sensor1)
Sensor2 = np.concatenate((data_Sensor2, locdatanp)); Sensor2 = np.transpose(Sensor2); SensorsH=np.concatenate((Sensor1, Sensor2))
Sensor3 = np.concatenate((data_Sensor3, locdatanp)); Sensor3 = np.transpose(Sensor3); SensorsH=np.concatenate((SensorsH, Sensor3))
Sensor4 = np.concatenate((data_Sensor4, locdatanp)); Sensor4 = np.transpose(Sensor4); SensorsH=np.concatenate((SensorsH, Sensor4))

###### Part M ####################################################################################################################

data_Sensor1, data_Sensor2, data_Sensor3, data_Sensor4 = ([] for i in range(4))  # Transient data
Sensor1, Sensor2, Sensor3, Sensor4 = ([] for j in range(4))  # Main body

for filename in os.listdir(mForce):  # Main body
    csv_file_path = os.path.join(mForce, filename)
    data = loadtxt(csv_file_path, delimiter=",")
    data = np.concatenate((data, SensorsIndex))
    data = data.transpose()
    data_Sensor1.append(data[0]); data_Sensor2.append(data[1]); data_Sensor3.append(data[2]); data_Sensor4.append(data[3])

# adding locations
data_Sensor1 = np.transpose(data_Sensor1); data_Sensor2 = np.transpose(data_Sensor2)
data_Sensor3 = np.transpose(data_Sensor3); data_Sensor4 = np.transpose(data_Sensor4)

Sensor1 = np.concatenate((data_Sensor1, locdatanp)); Sensor1 = np.transpose(Sensor1)
Sensor2 = np.concatenate((data_Sensor2, locdatanp)); Sensor2 = np.transpose(Sensor2); SensorsM=np.concatenate((Sensor1, Sensor2))
Sensor3 = np.concatenate((data_Sensor3, locdatanp)); Sensor3 = np.transpose(Sensor3); SensorsM=np.concatenate((SensorsM, Sensor3))
Sensor4 = np.concatenate((data_Sensor4, locdatanp)); Sensor4 = np.transpose(Sensor4); SensorsM=np.concatenate((SensorsM, Sensor4))

###### Part L ####################################################################################################################
data_Sensor1, data_Sensor2, data_Sensor3, data_Sensor4 = ([] for i in range(4))  # Transient data
Sensor1, Sensor2, Sensor3, Sensor4 = ([] for j in range(4))  # Main body

for filename in os.listdir(lForce):  # Main body
    csv_file_path = os.path.join(lForce, filename)
    data = loadtxt(csv_file_path, delimiter=",")
    data = np.concatenate((data, SensorsIndex))
    data = data.transpose()
    data_Sensor1.append(data[0]); data_Sensor2.append(data[1]); data_Sensor3.append(data[2]); data_Sensor4.append(data[3])

# adding locations
data_Sensor1 = np.transpose(data_Sensor1); data_Sensor2 = np.transpose(data_Sensor2)
data_Sensor3 = np.transpose(data_Sensor3); data_Sensor4 = np.transpose(data_Sensor4)

Sensor1 = np.concatenate((data_Sensor1, locdatanp)); Sensor1 = np.transpose(Sensor1)
Sensor2 = np.concatenate((data_Sensor2, locdatanp)); Sensor2 = np.transpose(Sensor2); SensorsL=np.concatenate((Sensor1, Sensor2))
Sensor3 = np.concatenate((data_Sensor3, locdatanp)); Sensor3 = np.transpose(Sensor3); SensorsL=np.concatenate((SensorsL, Sensor3))
Sensor4 = np.concatenate((data_Sensor4, locdatanp)); Sensor4 = np.transpose(Sensor4); SensorsL=np.concatenate((SensorsL, Sensor4))

# Add headers

AllData=np.concatenate((SensorsH, SensorsM)); AllData=np.concatenate((AllData, SensorsL))
print(AllData.shape)

hdrdata = pd.read_csv(sourcehdr, delimiter=",", header=None)
hdrdatalist = hdrdata[0].tolist()

df = pd.DataFrame(data=AllData, columns=hdrdatalist)
print(df.shape)

#Writing into CSV Files
df.to_csv(os.path.join(results, "MasterData.csv"), index=False)
