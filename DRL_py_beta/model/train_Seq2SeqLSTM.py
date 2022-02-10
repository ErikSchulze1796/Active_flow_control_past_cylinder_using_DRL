from os.path import isdir

import datetime

import torch
from torch import nn

def train_LSTM(encoder, decoder, train_data, val_data, save_best_dir: str, batch_size: int=64, n_epochs: int=1, learning_rate: float=0.001):
    """Train an LSTM as

    Parameters
    ----------
    model : [type]
        [description]
    train_data : [type]
        [description]
    val_data : [type]
        [description]
    save_best_dir : str
        [description]
    batch_size : int, optional
        [description], by default 64
    n_epochs : int, optional
        [description], by default 1
    learning_rate : float, optional
        [description], by default 0.001

    Returns
    -------
    [type]
        [description]
    """

    # Initialize train and validation dataloader for easy data handling with pytorch
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # Define loss function to MSE since its a regression task
    criterion = nn.MSELoss()
    # Optimize using adaptive momentum
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=learning_rate)
    # Initialize best loss variablesto a high value to check if next loss is lower
    best_val_loss, best_train_loss = 1.0e5, 1.0e5
    # Initialize arrays storing train and validation loss
    train_loss, val_loss = [], []

    # Get overall training start time
    t_start_training = datetime.datetime.now()
    # 1st loop: Loop over number of epochs
    for epoch in range(n_epochs):
        t_start_epoch = datetime.datetime.now()
        # Set model in train mode
        encoder.train()
        decoder.train()
        # Initialize accumulated epoch loss to 0.0
        acc_loss_train = 0.0
        acc_loss_val = 0.0

        # Get training loop start time
        t_start_train = datetime.datetime.now()
        # 2nd loop: Loop over batches
        for k, (feature_batch_train, label_batch_train) in enumerate(train_loader):
            # Sets the gradients of all optimized torch.Tensors to zero
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # Push features into encoder
            hidden = encoder(feature_batch_train)
            # Get prediction from decoder
            output_seq = decoder(label_batch_train, hidden)
            # Calculate training loss using the previously set optimizer (MSE)
            loss = criterion(output_seq, label_batch_train)
            # Backpropagate the loss and compute gradients
            loss.backward()
            # Optimize the weights
            encoder_optimizer.step()
            decoder_optimizer.step()
            # Accumulate loss over all batches to get average loss per epoch
            acc_loss_train += loss.item() * len(feature_batch_train)
        # Append current epoch training loss to training loss array
        train_loss.append(acc_loss_train/len(train_loader))
        t_end_train = datetime.datetime.now()
        t_train = (t_end_train-t_start_train).total_seconds()
        
        t_start_val = datetime.datetime.now()
        # Set model into evaluation mode, using the context manager is equivalent to "model.eval()"
        # Disables gradient calculation and reduces memory consumption and more
        # For more information please refer to https://pytorch.org/docs/stable/generated/torch.no_grad.html
        with torch.no_grad():
            # Validation batch loop
            for j, (feature_batch_val, label_batch_val) in enumerate(val_loader):
                # Push features into encoder
                hidden = encoder(feature_batch_val)
                # Get prediction from decoder
                output_seq = decoder(label_batch_val, hidden)
                # Calculate validation loss using the optimizer (MSE)
                loss = criterion(output_seq, label_batch_val.squeeze())
                # Accumulate validation loss over all batches to get average loss per epoch
                acc_loss_val += loss.item() * len(feature_batch_val)
            # Append current epoch validation loss to validation loss array
            val_loss.append(acc_loss_val/len(val_loader))
        # Get end times for epoch and validation
        t_end_epoch = datetime.datetime.now()
        t_val = (t_end_epoch-t_start_val).total_seconds()
        t_epoch = (t_end_epoch-t_start_epoch).total_seconds()
        t_elapsed = (t_end_epoch-t_start_training).total_seconds()
        # Print current trainnig and validation loss
        print("# =========================================== #")
        print("Epoch: {:5d}".format(epoch))
        print("Training loss: {:10.5e}".format(train_loss[-1]))
        print("Validation loss: {:10.5e}".format(val_loss[-1]))
        print("Computation times (t_elapsed|t_epoch|t_train|t_val): {} | {} | {} | {}".format(t_elapsed, t_epoch, t_train, t_val))
        
        # Get number of subsequent time steps used for prediction
        n_steps_history = next(iter(train_loader))[0].shape[1]
        # Save training and validation model to disk
        if isdir(save_best_dir):
            if train_loss[-1] < best_train_loss:
                torch.save(encoder.state_dict(), f"{save_best_dir}decoder_LSTM_best_model_train_{learning_rate}_n_history{n_steps_history}.pt")
                torch.save(decoder.state_dict(), f"{save_best_dir}encoder_LSTM_best_model_train_{learning_rate}_n_history{n_steps_history}.pt")
                best_train_loss = train_loss[-1]
                
            if val_loss[-1] < best_val_loss:
                torch.save(encoder.state_dict(), f"{save_best_dir}encoder_LSTM_best_model_val_{learning_rate}_n_history{n_steps_history}.pt")
                torch.save(decoder.state_dict(), f"{save_best_dir}decoder_LSTM_best_model_val_{learning_rate}_n_history{n_steps_history}.pt")
                best_val_loss = val_loss[-1]
        
        # Save training and validation loss to pytorch binary file
        torch.save(train_loss, f"{save_best_dir}LSTM_train_loss_{learning_rate}_n_history{n_steps_history}.pt")
        torch.save(val_loss, f"{save_best_dir}LSTM_val_loss_{learning_rate}_n_history{n_steps_history}.pt")

    t_training = (t_end_epoch-t_start_training).total_seconds()
    print("# =========================================== #")
    print("Training computation time : {}".format(t_training))
    return train_loss, val_loss
