# FFRecord

The FFRecord format is a simple format for storing a sequence of binary records developed by HFAiLab,
which supports random access and Linux Asynchronous Input/Output (AIO) read.

## File Format

**Storage Layout:**
```
+-----------------------------------+---------------------------------------+
|         checksum                  |             N                         |
+-----------------------------------+---------------------------------------+
|         checksums                 |           offsets                     |
+---------------------+---------------------+--------+----------------------+
|      sample 1       |      sample 2       | ....   |      sample N        |
+---------------------+---------------------+--------+----------------------+
```

**Fields:**

| field     | size (bytes)                  | description                     |
|-----------|-------------------------------|---------------------------------|
| checksum  | 4                             | CRC32 checksum of metadata      |
| N         | 8                             | number of samples               |
| checksums | 4 * N                         | CRC32 checksum of each sample   |
| offsets   | 8 * N                         | byte offset of each sample      |
| sample i  | offsets[i + 1] - offsets[i]   | data of the i-th sample         |

## Get Started

### Requirements

- OS: Linux
- Python >= 3.6
- Pytorch >= 1.6
- NumPy
- tqdm
- zlib: `sudo apt install zliblg-dev`
- cmake: `pip install cmake`
- pybind11 >= 2.8

### Install

```
python3 setup.py install
```

## Usage

We provide `ffrecord.FileWriter` and `ffrecord.FileReader` for reading and writing, respectively.

### Write

To create a `FileWriter` object, you need to specify a file name and the total number of samples.
And then you could call `FileWriter.write_one()` to write a sample to the FFRecord file.
It accepts `bytes` or `bytearray` as input and appends the data to the end of the opened file.

```python
from ffrecord import FileWriter


def serialize(sample):
    """ Serialize a sample to bytes or bytearray

    You could use anything you like to serialize the sample.
    Here we simply use pickle.dumps().
    """
    return pickle.dumps(sample)


samples = [i for i in range(100)]  # anything you would like to store
fname = 'test.ffr'
n = len(samples)  # number of samples to be written
writer = FileWriter(fname, n)

for i in range(n):
    data = serialize(samples[i])  # data should be bytes or bytearray
    writer.write_one(data)

writer.close()
```

### Read

To create a `FileReader` object, you only need to specify the file name.
And then you could call `FileWriter.read()` to read multiple samples from the FFReocrd file.
It accepts a list of indices as input and outputs the corresponding samples data.

The reader would validate the checksum before returning the data if `check_data = True`.

```python
from ffrecord import FileReader


def deserialize(data):
    """ deserialize bytes data

    The deserialize method should be paired with the serialize method above.
    """
    return pickle.loads(data)


fname = 'test.ffr'
reader = FileReader(fname, check_data=True)
print(f'Number of samples: {reader.n}')

indices = [3, 6, 0, 10]      # indices of each sample
data = reader.read(indices)  # return a list of bytes-like data

for i in range(n):
    sample = deserialize(data[i])
    # do what you want

reader.close()
```

### Dataset and DataLoader for PyTorch

We also provide `ffrecord.torch.Dataset` and `ffrecord.torch.DataLoader` for PyTorch users to train
models using FFRecord.

Different from `torch.utils.data.Dataset` which accepts an index as input and returns only one sample,
`ffrecord.torch.Dataset` accepts a batch of indices as input and returns a batch of samples.
One advantage of `ffrecord.torch.Dataset` is that it could read a batch of data at a time using Linux AIO.

Users need to inherit from `ffrecord.torch.Dataset` and define their custom `__getitem__()` and `__len__()` function.


For example:

```python
class CustomDataset(ffrecord.torch.Dataset):

    def __init__(self, fname, check_data=True, transform=None):
        self.reader = FileReader(fname, check_data)
        self.transform = transform

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        # we read a batch of samples at once
        assert isintance(indices, list)
        data = self.reader.read(indices)

        # deserialize data
        samples = [pickle.loads(b) for b in data]

        # transform data
        if self.transform:
            samples = [self.transform(s) for s in samples]
        return samples

dataset = CustomDataset('train.ffr')
indices = [3, 4, 1, 0]
samples = dataset[indices]
```

`ffrecord.torch.DataLoader` is a drop-in replacement for PyTorch's standard dataloader.
`ffrecord.torch.Dataset` could be combined with it just like PyTorch.
`ffrecord.torch.DataLoader` supports for skipping steps during training by `set_step()` method.

```python
dataset = CustomDataset('train.ffr')
loader = ffrecord.torch.DataLoader(dataset,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=8)

start_epoch = 5
start_step = 100  # resume from epoch 5, step 100
loader.set_step(start_step)

for epoch in range(start_epoch, epochs):
    for i, batch in enumerate(loader):
        # training model

    loader.set_step(0)  # remember to reset before the next epoch
```


### Pack a folder into ffrecord

FFRecord could also be used to pack a folder into a single file, which could be accessed without unpacking.

For example:

Assume we have a folder named `just_a_folder`:

```shell
$ tree just_a_folder

just_a_folder/
├── 001.txt
├── 002.txt
├── 003.txt
├── just_a_figure.png
└── just_another_folder
    ├── 004.txt
    ├── jsonfile.json
    ├── npyfile.npy
    ├── npzfile.npz
    └── another_folder
        └── 005.txt
```

Now we pack this folder into a file named `packed.ffr`:

```python
from ffrecord import pack_folder
pack_folder("just_a_folder", "packed.ffr", verbose=True)
```

And then we could access the packed folder by `PackedFolder`:

```python
>>> import io
>>> from ffrecord import PackedFolder
>>>
>>> folder = PackedFolder("packed.ffr")
>>> folder.list()
['001.txt', '002.txt', '003.txt', 'just_a_figure.png', 'just_another_folder']
>>> folder.list('just_another_folder')
['004.txt','jsonfile.json','npyfile.npy','npzfile.npz','another_folder']
>>> folder.is_file("just_another_folder")
False
>>> folder.is_dir("just_another_folder")
True
>>> folder.exists("just_another_folder/another_folder")
True
>>> fp = io.BytesIO(folder.read('001.txt'))
>>> data = fp.read()  # binary data
>>> list_of_data = folder.read(["001.txt", "002.txt"])  # read multiple files by Linux AIO
```

Here are some samples for reading file formats that are frequently used. Just replace your original code blocks with follows and enjoy FFRecord. 

#### Images：

```Python
import cv2
order = "RGB"
path = "just_a_figure.png"
fp = io.BytesIO(folder.read_one(path))
img = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
if order == 'RGB':
    img = img[:, :, ::-1].copy()
cv2.imwrite("test.png", img)
```

#### Texts:

```Python
fp = io.BytesIO(folder.read_one("just_another_folder/another_folder/005.txt"))
bytestring = fp.read()
result_str = bytestring.decode("utf-8")
print(result_str)
```

#### JSON：

```Python
import json
fp = io.BytesIO(folder.read_one("just_another_folder/jsonfile.json"))
bytestring = fp.read()
result_str = bytestring.decode("utf-8")
annot = json.loads(result_str)
print(annot)
```

#### Ndarrays saved in .npy file：

```Python
import numpy as np
fp = io.BytesIO(folder.read_one("just_another_folder/npyfile.npy"))
result = np.load(fp,allow_pickle=True)
print(result)
```

#### Ndarrays saved in .npz file:

```Python
# .npz file is a zip file for ndarrays, generated by np.savez
import numpy as np
import zipfile
fp = io.BytesIO(folder.read_one("just_another_folder/npzfile.npz"))
test = zipfile.ZipFile(fp,allowZip64=True)
print(test.namelist())
# arr_0 is a key in the namelist
with test.open('arr_0.npy',"r") as myfile:
    result = np.load(myfile,allow_pickle=True)
```

#### Write a file directly:

```Python
fp = io.BytesIO(folder.read_one("just_another_folder/npyfile.npy"))
with open("just_a_name", 'wb') as f:
    f.write(fp.read())
```
