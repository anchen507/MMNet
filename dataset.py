from data_utils import *
import matplotlib.pyplot as plt
from flow_utils import readFlow
class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(TrainSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size
        self.input_frame=input_frame
        with open(dataset_dir+'/sep_trainlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        #self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        FLOW=[]
        for i in range(self.input_frame):
            img_hr = Image.open(self.dir + '/sequences/' +  '/'+self.train_list[idx] + '/im' + str(i + 1) + '.png')  #'/sequences/' + 
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            if i!=(self.input_frame-1):
                flo = readFlow('./../../../../../../data2/anchen/data/opticflow/' +  '/'+self.train_list[idx] + '/im' + str(i + 1) + '.flo')
                flo = flo.transpose(2,0,1)
                FLOW.append(flo)
            HR.append(img_hr)
        HR = np.stack(HR, 1)
        FLOW.append(flo)# for augumentation 
        FLOW = np.stack(FLOW, 1)
        combine_data=np.concatenate((HR,FLOW),axis=0)
        combine_data = random_crop(combine_data, self.patch_size)

        #combine_data = self.tranform(combine_data)

        combine_data = torch.from_numpy(np.ascontiguousarray(combine_data))

        HR = combine_data[0:3,:,:,:]
        #print(np.shape(HR))     
        FLOW = combine_data[3:5,0:self.input_frame-1,:,:]
        #print(np.shape(FLOW))
        return  HR, FLOW
    def __len__(self):
        return len(self.train_list)

class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(TrainSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size
        self.input_frame=input_frame
        with open(dataset_dir+'/sep_testlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        #self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        for i in range(self.input_frame):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            HR.append(img_hr)

        HR = np.stack(HR, 1)

        HR = torch.from_numpy(np.ascontiguousarray(HR))

        return HR

    def __len__(self):
        return len(self.train_list)
class TestSetLoader(Dataset):
    def __init__(self, dataset_dir,patch_size, input_frame):
        super(TestSetLoader).__init__()
        self.dir = dataset_dir
        self.patch_size=patch_size

        self.input_frame=input_frame
        with open(dataset_dir+'/sep_testlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        #self.tranform = augumentation()
    def __getitem__(self, idx):
        HR = []
        for i in range(7):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_hr = img_hr.transpose(2,0,1)
            HR.append(img_hr)

        HR = np.stack(HR, 1)

        HR = torch.from_numpy(np.ascontiguousarray(HR))

        return HR

    def __len__(self):
        return len(self.train_list)


class augumentation(object):
    def __call__(self, input):
        if random.random()<0.5:
            input = input[::-1, :, :]
        if random.random()<0.5:
            input = input[:, ::-1, :]
        if random.random()<0.5:
            input = input.transpose(0, 1, 3, 2)#C N H W
        return input
