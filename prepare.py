import os
import random
import shutil


template = './template'
data = './test'
test_l = os.listdir(data)
for i in test_l:
    temp_path = os.path.join(template,i)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path,777)
    testdata = os.path.join(data,i)
    file = os.listdir(testdata)
    seed = random.randint(0,len(file)-1)
    test_p = os.path.join(testdata,file[seed])
    temp_p = os.path.join(temp_path,file[seed])
    shutil.copyfile(test_p,temp_p)
