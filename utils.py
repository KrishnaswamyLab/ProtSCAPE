import os
import numpy as np

from argparse import ArgumentParser
from gsae_model import GSAE
import pytorch_lightning as pl
import torch
from de_shaw_Dataset import DEShaw, Scattering
from tqdm import tqdm


parser = ArgumentParser()

parser.add_argument('--dataset', default='deshaw', type=str)

parser.add_argument('--input_dim', default=None, type=int)
parser.add_argument('--bottle_dim', default=25, type=int)
parser.add_argument('--hidden_dim', default=400, type=int)
parser.add_argument('--learning_rate', default=0.00001, type=float)

parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--beta', default=0.0005, type=float)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--len_epoch', default=None)

parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--n_gpus', default=1, type=int)
parser.add_argument('--save_dir', default='train_logs/', type=str)

# add args from trainer
parser = pl.Trainer.add_argparse_args(parser)
# parse params 
args = parser.parse_args()
full_dataset = DEShaw('graphs/total_graphs.pkl', transform=Scattering())

args.input_dim = len(full_dataset[0][0])

model = GSAE(input_dim=args.input_dim, bottle_dim=args.bottle_dim, hidden_dim=args.hidden_dim,\
                    learning_rate=args.learning_rate, alpha=args.alpha, beta=args.beta, n_epochs=args.n_epochs,\
                    len_epoch=args.len_epoch, batch_size=args.batch_size, n_gpus=args.n_gpus, save_dir=args.save_dir)

model.load_state_dict(torch.load("model.npy"))
model.eval()


scats = []
times = []
for i in tqdm(range(len(full_dataset))):
    scats.append(full_dataset[i][0].numpy())
    times.append(full_dataset[i][1][0].numpy())





np.save("scat_coeffs.npy", scats)
np.save("times.npy", times)
scats = np.array(scats)
scats = torch.tensor(scats)


with torch.no_grad():
    #loss = model.get_loss_list()
    ##    train_embed = model.embed(train_tup[0])[0]
    #    test_embed =  model.embed(test_tup[0])[0]

    embed = model.embed(scats)[0]

np.save("embeddings.npy", embed)


# save_arr = []
# arr = os.listdir("/data/lab/de_shaw/all_trajectory_slices/GB3/8 to 10 us")
# for entry in arr:
#     save_arr.append(entry)

# np.save("file_names/pdgs8to10.npy", save_arr)

