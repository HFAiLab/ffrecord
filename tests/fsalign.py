import os
from ffrecord import checkFsAlign

fd = os.open("/public_dataset/2/VOCtrainval_11-May-2012.tar",
             os.O_RDONLY | os.O_DIRECT)
# fd = os.open("/fs-jd/prod/private/zsy/test_fs", os.O_RDONLY | os.O_DIRECT)
print(checkFsAlign(fd))