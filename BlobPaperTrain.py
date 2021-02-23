import os, math, sys, json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from scipy.ndimage import label
from BlobDetector import BlobDetector
from KeypointDataset import Keypoint_Dataset
from lcfcn import lcfcn_loss

def get_model():
    return BlobDetector()
 
# Hyperparameters and other arguments are now stored in a JSON file
def get_args(file_name='config.json'):
    with open(file_name) as config_file:
        data = json.load(config_file)
        return data
    return None
    
# Validation set loader is not necessary, and the verbose flag is for how much info you want
def fit(model, device, loader, val_loader=None, verbose='', epochs=5, learning_rate=1e-8, momentum=0.7,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3, save_path=None):

    losses = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, 
                                weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    model.train()
    for epoch in range(epochs):
        train_count = 1
        epoch_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            batch_size = imgs.shape[0]

            preds = model(imgs)
            points = masks.squeeze()
            loss = lcfcn_loss.compute_loss(points=points, probs=preds.sigmoid())
            
            loss.backward()
            optimizer.step()
            avg_loss = loss.mean()
            epoch_loss += avg_loss 
            
            if verbose:
                print('Loss:', loss)
            
            # nan check
            if math.isnan(avg_loss.item()):
                print('\n',loss, '\n')
                print(f'dataset item #{epoch}: {loader.dataset.__getitem__(epoch)[1]}')
                raise Exception('something is fucked')

            train_count += 1
        print(f'Train: Epoch #{epoch+1} total loss: {epoch_loss/train_count}')

        if val_loader is not None:
            epoch_loss = 0
            val_count = 1
            abs_err = 0
            best_err = float('inf')
            with torch.no_grad():
                for imgs, masks in loader:
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    preds = model(imgs).sigmoid()
                    points = masks.squeeze()
                    val_loss = lcfcn_loss.compute_loss(points=points, probs=preds)
                    
                    true_count = label(points.cpu())[-1]
                    pred_count = label(preds.cpu().ge(0.5))[-1]
                    abs_err += abs(true_count - pred_count)
                    
                    avg_loss = val_loss.mean()
                    epoch_loss += avg_loss
                    
                    if verbose:
                        print('Loss:', loss)
                    val_count += 1
            print(f'Val: Epoch #{epoch+1} total loss: {epoch_loss / val_count}')

            if abs_err < best_err and save_path is not None:
                # Save the model
                torch.save(model.state_dict(), save_path)


        lr_scheduler.step()
    return losses



def get_loaders(data_split_ratio, batch_size, files):
    
    tif = files['tif'] #'/research/wehrwein/wallin/blaine_harbor/blaine_June12/Blaine_June12_2019_flt2_3_ortho.tif'
    shp_file = files['shp_file'] #'/research/wehrwein/wallin/blaine_harbor/blaine_June12/comorant_nest_6_12.shp'
    img_dir = files['img_dir'] #'/research/wehrwein/wallin/blaine_harbor/sliced_pics/Blaine_June12_2019_flt2_3_ortho[1000]'

    val_tif = '/research/wehrwein/wallin/blaine_harbor/blaine_June19/Blaine_June19_2019_flt1_2_ortho.tif'
    val_shp = '/research/wehrwein/wallin/blaine_harbor/blaine_June19/cormorant_6_19.shp'
    val_img_dir = '/research/wehrwein/wallin/blaine_harbor/sliced_pics/Blaine_June19_2019_flt1_2_ortho'
    trans = T.Compose([T.ToPILImage(), T.Resize(224), T.ColorJitter(), T.ToTensor()])
    
    train_set = Keypoint_Dataset(tif, shp_file, img_dir, transforms=trans)
    
    # Get the total amount of training data from the dataset
    total = train_set.data
    
    # Split the data by the split_ratio and give the first part of the split to train
    train_data = total[:int(len(train_set) * data_split_ratio)]
    train_set.data = train_data
    
    # Give second part to val
    val_data = train_data[int(len(train_set) * (1 - data_split_ratio)):]
    val_set = Keypoint_Dataset(tif, shp_file, img_dir, data=val_data, transforms=trans)

    t_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print('Loaders ready')
    return t_loader, val_loader


def main(config_file='config.json'):
    args = get_args(config_file)

    # Runs on gpu if it is available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Creates a new model to predict the given classes
    model = get_model().to(device)

    # Data split is for splitting test data and validation data (e.g 0.8 is 80% train, 20% val)
    train_loader, val_loader = get_loaders(args["data_split"], args["batch_size"], args["files"])

    save_path = Path('../models/paths', args["name"])

    # Train and validate the model with our hyperparameters
    fit(model, device, train_loader, val_loader, verbose=args["verbose"], epochs=args["epochs"], 
        learning_rate=args["lr"],lr_step_size=args["lr_step"], momentum=args["momentum"], save_path=save_path)

   


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(config_file=sys.argv[1])
    else:
        main()
