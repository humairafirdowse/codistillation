import pandas as pd
import numpy as np
from keras.preprocessing import image
import torch
import keras
import itertools
import torch
from PIL import Image
import torchvision.transforms as transforms
import os


np.random.seed(0)

def celebA_loader(corr,data_length,n_clients):
    df = pd.read_csv('./celeb/anno.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    
    task_list = ["High_Cheekbones", "Mouth_Slightly_Open","Smiling","5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","Male","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
    task_list = task_list[:n_clients]
    df = df[["q", "file_name"]+ task_list]
    t_df = df


    for i in range(n_clients):
        t_df = t_df.loc[(df[task_list[i]] == 0)]
    
    new_df = t_df.head(n=int(data_length*corr/2))
    
    t_df = df

    for i in range(n_clients):
        t_df = t_df.loc[(df[task_list[i]] == 1)]
    
    t_df = t_df.head(n=int(data_length*corr/2))
    new_df = new_df.append(t_df)

    lst = [list(i) for i in itertools.product([0, 1], repeat=n_clients)]
    lst = lst[1:-1]
    rest = len(lst)
    # print(rest)
    for per in lst:
        t_df = df
        for i in range(n_clients):
            t_df = t_df.loc[(df[task_list[i]] == per[i])]
        t_df = t_df.head(n=int((data_length - int(data_length*corr))/rest))
        new_df = new_df.append(t_df)

    df = new_df.sample(frac=1).reset_index(drop=True)
    train_image = []
    for i in range(data_length):
        img = keras.utils.load_img('./celeb/img_align_celeba/'+df['file_name'][i],target_size=(178,218,3))
        img = keras.utils.img_to_array(img)
        img = img/255
        train_image.append(img.T)
    X = np.array(train_image)

    df = df.head(n=data_length)
    y = np.array(df.drop(['q', 'file_name'],axis=1))
    y = y.astype('float32')
    y = torch.from_numpy(y)
    y = y.long()
    X = torch.from_numpy(X)

    return X, y


def get_covid(_type_,_size_):
    df = pd.DataFrame(columns = ['image', 'label'])
    
    if(_type_ != "train"):
        _size_ *= 1 
        path_pos="/content/Dataset/Client-A/testing/Positive/"
        path_neg="/content/Dataset/Client-A/testing/Negative/" 
        no_images = [1341-_size_, 1341-_size_]
    else:
        path_pos="/content/Dataset/Client-A/training/Positive/"
        path_neg="/content/Dataset/Client-A/training/Negative/" 
        no_images = [_size_, _size_]  
    
    act_no = [_size_*600]*2
    label = 0
    permu = np.random.permutation(no_images[label])
    i = 0
    while i < min(act_no[label], no_images[label]):
        df = df.append({'image' : f'{path_neg}{permu[i]}.png', 'label' : label}, ignore_index = True)
        i += 1
    label = 1
    i = 0
    print(len(df),"lendf1")
    permu = np.random.permutation(no_images[label])
    while i < min(act_no[label], no_images[label]):
        df = df.append({'image' : f'{path_pos}{permu[i]}.png', 'label' : label}, ignore_index = True)
        i += 1
    i = 0
    
    images = []
    labels = []
    GRAY = transforms.Grayscale()
    transform = transforms.Compose([transforms.PILToTensor()])
    for index, row in df.iterrows():
        image = Image.open(row['image'])
        image = image.resize((200,200))
        image = GRAY(image)
        img_tensor = transform(image).to(torch.float)
        if(img_tensor.shape[0]!=1):
            print(img_tensor.shape)
            continue
        images.append(img_tensor)
        labels.append(row['label'])
    labels = torch.tensor(labels,dtype=torch.long)
    images = torch.stack(images, dim=0)
    return images, labels

def get_ocular_(_type_,_size_):
    df = pd.DataFrame(columns = ['image', 'label'])
    # normal, diabetes, hypertension
    no_images = [2874, 80, 192]
    if(_type_ != "train"):
        _size_ *= 1    
    
    act_no = [_size_]*4
    entries = os.listdir('/home/pcvishak/Multi-Fed/ocular/Normal/images/')
    dir_normal='/home/pcvishak/Multi-Fed/ocular/Normal/images/'
    label=0
    print(len(entries))
    for image in entries[:act_no[label]]:
        df = df.append({'image' : dir_normal+image, 'label' : label}, ignore_index = True)

    entries = os.listdir('/home/pcvishak/Multi-Fed/ocular/Diabetes/images/')
    dir_diabetes='/home/pcvishak/Multi-Fed/ocular/Diabetes/images/'
    label=1
    print(len(entries))
    for image in entries[:act_no[label]]:
        df = df.append({'image' : dir_diabetes+image, 'label' : label}, ignore_index = True)

    entries = os.listdir('/home/pcvishak/Multi-Fed/ocular/Hypertension/images/')
    dir_hypertension='/home/pcvishak/Multi-Fed/ocular/Hypertension/images/'
    label=2
    print(len(entries))
    for image in entries[:act_no[label]]:
        df = df.append({'image' : dir_hypertension+image, 'label' : label}, ignore_index = True)

    images = []
    labels = []
    GRAY = transforms.Grayscale()
    transform = transforms.Compose([transforms.PILToTensor()])
    for index, row in df.iterrows():
        image = Image.open(row['image'])
        image = image.resize((200,200))
        image = GRAY(image)
        img_tensor = transform(image).to(torch.float)
        if(img_tensor.shape[0]!=1):
            print(img_tensor.shape)
            continue
        images.append(img_tensor)
        labels.append(row['label'])
    labels = torch.tensor(labels,dtype=torch.long)
    images = torch.stack(images, dim=0)
    return images, labels

def get_der(_size_):
    data = np.load('Data.npz')
    X_train = torch.from_numpy(np.concatenate((data['train_images'],data['val_images']), axis=0))
    X_test = torch.from_numpy(data['test_images'])
    Y_train = torch.squeeze(torch.from_numpy(np.concatenate((data['train_labels'],data['val_labels']), axis=0)))
    Y_test = torch.squeeze(torch.from_numpy(data['test_labels']))
    
    train_X = []
    test_X  = []
    train_Y = []
    test_Y  = []
    
    for i, cls_ in enumerate([2,1,0]):
        idx_train = (Y_train==cls_).nonzero().squeeze()
        # print(idx_train.shape)
        idx_train = idx_train[:_size_] 
        Y_train[idx_train] = i
        train_X.append(X_train[idx_train])
        train_Y.append(Y_train[idx_train])
        idx_test = (Y_test==cls_).nonzero().squeeze() 
        Y_test[idx_test] = i
        test_X.append(X_test[idx_test])
        test_Y.append(Y_test[idx_test])
    train_Y = torch.cat(train_Y, axis=0)
    train_X = torch.cat(train_X, axis=0).permute(0, 3, 1, 2)
    test_X = torch.cat(test_X, axis=0).permute(0, 3, 1, 2)
    test_Y = torch.cat(test_Y, axis=0)

    return train_X, train_Y.long(), test_X, test_Y.long() 




def get_ocular(_size_):
    df_full = pd.read_csv('./ocu/full_df.csv')
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    classes_ = ["['N']","['D']","['G']"]
    transform = transforms.Compose([transforms.PILToTensor()])
    for L, cls_ in enumerate(classes_):
        df = df_full[df_full['labels'].isin([cls_])]
        df = df.head(_size_)
        df = df.reset_index()
        for idx, row in df.iterrows():
            image = Image.open(f'./ocu/ODIR-5K/ODIR-5K/Training Images/'+row['filename'])
            image = image.resize((200,200))
            img_tensor = transform(image).to(torch.float)
            if(idx < 0.8 * _size_):
                train_images.append(img_tensor)
                train_labels.append(L)
            else:
                test_images.append(img_tensor)
                test_labels.append(L)
    train_labels = torch.tensor(train_labels,dtype=torch.long)
    train_images = torch.stack(train_images, dim=0)
    
    test_labels = torch.tensor(test_labels,dtype=torch.long)
    test_images = torch.stack(test_images, dim=0)
    
    return train_images, train_labels, test_images, test_labels