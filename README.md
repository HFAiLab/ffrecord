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
| checksum  | 4                             | CRC32 checksum of all data      |
| N         | 8                             | number of samples               |
| checksums | 4 * N                         | CRC32 checksum of each sample   |
| offsets   | 8 * N                         | byte offset of each sample      |
| sample i  | offsets[i + 1] - offsets[i]   | data of the i-th sample         |

## Get Started

### Requirements

- OS: Linux
- Python >= 3.6
- Pytorch >= 1.6
- [libaio 0.8.3](https://pypi.org/project/libaio/)

### Install

```
pip3 install ffrecord
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
data = reader.read(indices)  # return a list of bytes data

for i in range(n):
    sample = deserialize(data[i])
    # do what you want

reader.close()
```

### Dataset and DataLoader for PyTorch

We also provide `ffrecord.torch.Dataset` and `ffrecord.torch.DataLoader` for PyTorch users to train
models using FFRecord.

Different from `torch.utils.data.Dataset` which accepts an index as input and returns one sample,
`ffrecord.torch.Dataset` accepts a batch of indices as input and returns a batch of samples.
One advantage of `ffrecord.torch.Dataset` is that it could read a batch of data at a time using Linux AIO.

We first read a batch of bytes data from the FFReocrd file and then pass the bytes data to `process()`
function. Users need to inherit from `ffrecord.torch.Dataset` and define their custom `process()` function.

```
Pipline:   indices ----------------------------> bytes -------------> samples
                      reader.read(indices)               process()
```

For example:

```python
class CustomDataset(ffrecord.torch.Dataset):

    def __init__(self, fname, check_data=True, transform=None):
        super().__init__(fname, check_data)
        self.transform = transform

    def process(self, indices, data):
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

`ffrecord.torch.Dataset` could be combined with `ffrecord.torch.DataLoader` just like PyTorch.

```python
dataset = CustomDataset('train.ffr')
loader = ffrecord.torch.DataLoader(dataset,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=8)

for i, batch in enumerate(loader):
    # training model

```
