import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from Dataset import EcologyDataset
from globalSumPool import ObjectCounter
from detecto import core, utils
import math
from utils import new_collate, IoU_and_Correct
from pathlib import Path
import os, sys
sys.path.append('../')
from data_prep.GSP_dataset import GSP_Dataset


def get_model():
    return ObjectCounter()


# Validation set loader is not necessary, and the verbose flag is for how much info you want
def fit(model, loader, val_loader=None, verbose=False, epochs=5, learning_rate=1e-8, momentum=0.7,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3):

    losses = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, 
                                weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_loss_dict = {}
        train_count = 1
        total_loss = 0
        for images, targets in loader:
            loss_dict = model(images, targets)

            for loss_type in loss_dict.keys():
                if loss_type in epoch_loss_dict:
                    epoch_loss_dict[loss_type] += loss_dict[loss_type]
                else:
                    epoch_loss_dict[loss_type] = loss_dict[loss_type]


            total_loss = sum(loss for loss in loss_dict.values())

            epoch_loss += total_loss

            if verbose:
                print('train:', total_loss)
            
            # nan check
            if math.isnan(total_loss.item()):
                print('\n',loss_dict, '\n')
                print(f'dataset item #{epoch}: {loader.dataset.__getitem__(epoch)[1]}')
                raise Exception('something is fucked')

            train_count += 1
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        loss_types = [f'{l_type}: {(loss/epoch+1)}' for l_type, loss in epoch_loss_dict.items()]
        print(f'Train: Epoch #{epoch+1} loss types: {loss_types}')
        print(f'Train: Epoch #{epoch+1} total loss: {epoch_loss/(epoch+1)}')

        if val_loader is not None:
            val_dict = {}
            val_loss = 0
            epoch_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    loss_dict = model(images, targets)

                    for loss_type in loss_dict.keys():
                        if loss_type in val_dict:
                            val_dict[loss_type] += loss_dict[loss_type]
                        else:
                            val_dict[loss_type] = loss_dict[loss_type]

                    val_loss = sum(loss for loss in loss_dict.values())
                    epoch_loss += val_loss

                if verbose:
                    print('val:', val_loss)
            
                losses.append(epoch_loss/(epoch+1))

                loss_types = [f'{l_type}: {(loss/epoch+1)}' for l_type, loss in val_dict.items()]
                print(f'Val: Epoch #{epoch+1} loss types: {loss_types}')
                print(f'Val: Epoch #{epoch+1} total loss: {epoch_loss/(epoch+1)}')




        lr_scheduler.step()
    return losses



def get_loaders(data_split_ratio, batch_size):
    # only one tif file works right now (flt_2_3_ortho) and I don't know why
    tif = '/research/wehrwein/wallin/blaine_harbor/blaine_June12/Blaine_June12_2019_flt2_3_ortho.tif'
    shp_file = '/research/wehrwein/wallin/blaine_harbor/blaine_June12/comorant_nest_6_12.shp'
    img_dir = '/research/wehrwein/wallin/blaine_harbor/sliced_pics/Blaine_June12_2019_flt2_3_ortho'

    val_tif = '/research/wehrwein/wallin/blaine_harbor/blaine_June19/Blaine_June19_2019_flt1_2_ortho.tif'
    val_shp = '/research/wehrwein/wallin/blaine_harbor/blaine_June19/cormorant_6_19.shp'
    val_img_dir = '/research/wehrwein/wallin/blaine_harbor/sliced_pics/Blaine_June19_2019_flt1_2_ortho'
    
    
    
    train_set = GSP_Dataset(tif, shp_file, img_dir)
    # Get the total amount of training data from the dataset
    total = train_set.data
    # Split the data by the split_ratio and give the first part of the split to train
    train_data = total[:int(train_set.__len__() * data_split_ratio)]
    train_set.data = train_data
    # Give second part to val
    val_data = train_data[int(train_set.__len__() * (1 - data_split_ratio)):]
    val_set = GSP_Dataset(tif, shp_file, img_dir, data=val_data)

    t_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=new_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=new_collate)
    print('Loaders ready')
    return t_loader, val_loader

def main(verbose=False, data_split=0.8, batch_size=1, epochs=100, lr=1e-4, 
         lr_step=3, name='gsp_v1.pth'):
    # Runs on gpu if it is available 
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    # Creates a new model to predict the given classes
    model = get_model().to(device)

    # Data split is for splitting test data and validation data (e.g 0.8 is 80% train, 20% val)
    train_loader, val_loader = get_loaders(data_split, batch_size)

    # Train and validate the model with our hyperparameters
    fit(model, train_loader, val_loader, epochs=epochs, learning_rate=lr, lr_step_size=lr_step, verbose=verbose)

    # Save the model
    torch.save(model.state_dict(), Path(os.getcwd(), name))

if __name__ == '__main__':
    main()
