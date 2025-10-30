import pandas as pd
import numpy as np
import torch 
from torch.utils.data import DataLoader,Dataset


class Processer:
    def __init__(self):
        # 将fit和transform需要的变量保存在成员变量里
        self.sex_mapping = None
        self.embarked_mapping = None
        self.age_median = None
        self.embarked = None 

    def fit(self,df):
        # 遍历元素，学习对应规则
        self.sex_mapping = {v:i for i,v in enumerate(df["Sex"].unique())}
        self.embarked_mapping = {v:i for i,v in enumerate(df["Embarked"].unique())}
        # 学习规则，补充缺失值
        self.age_median = df["Age"].median()
        self.embarked = df["Embarked"].mode()[0]
        # fit之后所有的值都被赋值给了实例，才能执行接下来的transform操作
    def transform(self,df):
        df = df.copy()
        # 删除列
        df = df.drop(columns=["PassengerId","Name","Ticket","Cabin"])
        # 补全值
        df["Age"] =  df["Age"].fillna(self.age_median)
        df["Embarked"] = df["Embarked"].fillna(self.embarked)

        df["Sex"] = df["Sex"].map(self.sex_mapping)
        df["Embarked"] = df["Embarked"].map(self.embarked_mapping)

        return df

class TitanicSet(Dataset):
    def __init__(self,data,is_train=True):
        self.is_train = is_train
        if is_train:
            # 拿了整个数据
            self.y = torch.from_numpy(data[:,[0]]).float()
            self.x = torch.from_numpy(np.delete(data, 0, axis=1)).float()
        else:
            self.x = torch.from_numpy(data).float()
            self.y = None   
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # getitem只需要返回指定位置的数据就可以
        if self.is_train:
            return self.x[index],self.y[index]
        else:
            return self.x[index]

class TitanicModel(torch.nn.Module):
    def __init__(self):
        super(TitanicModel,self).__init__()
        self.linear1 = torch.nn.Linear(7,4)
        self.linear2 = torch.nn.Linear(4,2)
        self.linear3 = torch.nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

if __name__ == '__main__':
    # data process
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    print(train_set.info())
    p = Processer()
    p.fit(train_set)
    train_set = p.transform(train_set)
    test_set = p.transform(test_set)
    # data load
    train_set = TitanicSet(train_set.values,is_train=True)
    test_set = TitanicSet(test_set.values,is_train=False)
    # 最后的数据封装在打他loader上完成
    train_loader = DataLoader(dataset=train_set,batch_size=4,shuffle=True,num_workers=2)

    # model & loss & optim
    model = TitanicModel()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    # training cycle
    for epoch in range(100):
        # 在training cycle上直接遍历打他loader的实例，返回值格式参照getitem
        for i,data in enumerate(train_loader): #每次遍历都会调用get item
            inputs,labels = data
            # forward
            y_hat = model(inputs)
            # gradient 
            loss = loss_function(y_hat,labels)
            print(epoch,i,loss.item())
            # backward 
            optimizer.zero_grad()
            loss.backward()
            # update 
            optimizer.step()

    torch.save(model.state_dict(), 'titanic_model.pth')
    print("模型已保存到 titanic_model.pth")



