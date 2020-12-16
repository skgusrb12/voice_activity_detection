import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
# from torchsummary import summary as summary_

from data_tools import prepare_input_img, check_length
import model_nn as model


def train_model(params):

    path = params['path']
    model_path = params['model_path']
    batch_size = params['batch_size']
    num_epochs = params['epochs']
    train_val_ratio = params['train_val_ratio']
    early_stopping = params['baseline_val_loss']

    data = np.expand_dims(np.load(os.path.join(path, 'train_data.npy')), 1)
    label = np.load(os.path.join(path, 'train_label.npy'))

    train_loader, val_loader = prepare_input_img(data, label, train_val_ratio, batch_size)
    num_total_batch, str_total_batch, str_epochs = check_length(train_loader, num_epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = model.Resnet1D(params).to(device)

    # check the structure of the model
    # summary_(net, (1, 65, 16), batch_size=32)

    train_losses, valid_losses = [], []
    avg_train_losses, avg_valid_losses = [], []
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    print('\nstart the training\n')

    for epoch in range(1, num_epochs+1):

        # training step
        net.train()
        for batch, (data, target) in enumerate(train_loader, 1):

            optimizer.zero_grad()

            out = net(data.to(device))
            loss = criterion(out, target.to(device))

            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

            print_batch_msg = (f'\r[batch : {batch:>{str_total_batch}}/{num_total_batch:>{str_total_batch}} ]')
            print(print_batch_msg, end=' ')

        # validation step
        total, correct = 0, 0
        net.eval()
        for data, target in val_loader:
            target = target.to(device)

            out = net(data.to(device))
            loss = criterion(out, target)

            valid_losses.append(loss.item())

            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print_loss_msg = (f'\r[{epoch:>{str_epochs}}/{num_epochs:>{str_epochs}} ]' +
                     f' train_loss: {train_loss:.5f} ' +
                     f'/ valid_loss: {valid_loss:.5f}' +
                     f'/ valid_acc: {100 * correct / total:.3f}')

        print(print_loss_msg)

        # Early Stopping
        if valid_loss < early_stopping:
            print('Early stopping!!!')
            break

    # save the weights and the model
    save_fn = os.path.join(model_path, "vad_model.pt")
    torch.save(net, save_fn)
    print('\nsaving the model')







