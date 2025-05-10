"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Train ViT
    Train a ViT to classify images
    Periodically output training information, and saves model checkpoints
    Usage: python train_vit.py
"""

import torch
import matplotlib.pyplot as plt

from dataset import get_train_val_test_loaders
from model.vit import ViT
from train_common import count_parameters, restore_checkpoint, evaluate_epoch, train_epoch, save_checkpoint, early_stopping
from utils import config, make_training_plot, set_random_seed


def main():
    """Train ViT and show training plots."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("vit.batch_size"),
    )

    # Define the ViT Model according to the appendix D
    model = ViT(
        num_patches=16,
        num_blocks=2,
        num_hidden=16,
        num_heads=2,
        num_classes=2,
        chw_shape=(3, 64, 64)
    )

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )
    
    print(f"Number of float-valued parameters: {count_parameters(model)}")

    # Attempts to restore the latest checkpoint if exists
    print("Loading ViT...")
    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))
    
    start_epoch = 0
    stats = []    

    axes = make_training_plot(name="ViT Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # define patience for early stopping
    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
        )

        save_checkpoint(model, epoch + 1, config("vit.checkpoint"), stats)

        # Update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")

    # Save figure and keep plot open; for debugging
    plt.savefig(f"vit_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()
    # raise NotImplementedError()


if __name__ == "__main__":
    main()
