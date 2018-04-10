import uwb_dataset
import numpy as np

# import raw data
data = uwb_dataset.import_from_files()

# divide CIR by RX preable count (get CIR of single preamble pulse)
# item[2] represents number of acquired preamble symbols
for item in data:
    item[15:] = item[15:]/float(item[2])

# test data
train = data[:30000, :]
np.random.shuffle(train)
x_train = train[:30000, 15:]
y_train = train[:30000, 0]
x_test = data[30000:, 15:]
y_test = data[30000:, 0]

# feed data
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

# feed data
x_val = x_train[25000:]
y_val = y_train[25000:]
x_train = x_train[:25000]
y_train = y_train[:25000]

Nnew=[]
for item in x_train:
    item = item[max([0,item.argmax()-50]) : item.argmax()+50]
    Nnew.append(item)
x_train = np.asarray(Nnew)
print(x_train.shape)
