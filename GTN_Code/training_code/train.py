import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler
from dataset import MoleculeDatasetWrapper
from gtn import GTN
import argparse
import matplotlib.pyplot as plt
from nt_xent import NTXentLoss
from tqdm import tqdm
from utils import init_seed

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    init_seed(seed=555)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GTN',
                        help='Model')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Training Epochs')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers')
    parser.add_argument('--valid', type=float, default=0.05,
                        help='data size for validation')
    parser.add_argument('--path', type=str, default='/data1/gx/GTN_Code/dataset/pubchem-10m-clean.txt',
                        help='dataset path') 
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GT/FastGT layers')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    batch_size = args.batch
    num_workers = args.num_workers
    valid_size = args.valid
    data_path = args.path
    node_dim = args.node_dim
    emb_dim = args.emb_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    dataset = MoleculeDatasetWrapper(batch_size, num_workers, valid_size, data_path)
    train_loader, valid_loader = dataset.get_data_loaders()

    model = GTN(
        num_channels = num_channels,
        w_in = emb_dim,
        w_out = node_dim,
        num_layers = num_layers,
        args = args,
        emb_dim = emb_dim
    )

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.to(device)
    model.train()
    loss_func = NTXentLoss(device, batch_size, temperature = 0.1, use_cosine_similarity = True)

    train_loss = []
    for epoch in range(epochs):
        # training
        total_loss = 0
        for bn, (xis, xjs) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            xis = xis.to(device)
            xjs = xjs.to(device)

            with autocast():
                ris, zis = model(xis, eval=eval)
                rjs, zjs = model(xjs, eval=eval)

                F.normalize(zis, dim=1)
                F.normalize(zjs, dim=1)

                loss = loss_func(zis, zjs)

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

        avg_train_loss = total_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        print(f'epoch {epoch+1} finished!')
        print(f'training loss is :{avg_train_loss}')

        # validation
        minloss = 999
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                valid_loss = 0

                for (xis, xjs) in valid_loader:
                    xis = xis.to(device)
                    xjs = xjs.to(device)

                    with autocast():
                        ris, zis = model(xis, eval=eval)
                        rjs, zjs = model(xjs, eval=eval)

                        F.normalize(zis, dim=1)
                        F.normalize(zjs, dim=1)

                        loss = loss_func(zis, zjs)

                    valid_loss += loss.item()                                 
            avg_val_loss = valid_loss / len(valid_loader)
            print(f'validation loss is :{avg_val_loss}')

            if avg_val_loss < minloss:
                minloss = avg_val_loss
                torch.save(model.state_dict(), 'model.pth')
            model.train()
    print('min validation loss is: ', minloss)

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss on pretrain')
    plt.legend()
    plt.savefig('pretrain_loss_model.png')