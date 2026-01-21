import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()

    loop = tqdm(data_loader, leave=True)
    losses = []
    for batch_idx, (masked, mask, target) in enumerate(loop):
        masked = masked.to(device)
        mask = mask.to(device)
        target = target.to(device)

        output = model(masked)
        loss = loss_fn(output[mask], target[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    loop.set_postfix(loss=np.mean(losses))
    return np.mean(losses)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    loop = tqdm(data_loader, leave=True)
    losses = []
    for batch_idx, (masked, mask, target) in enumerate(loop):
        masked = masked.to(device)
        mask = mask.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(masked)
            loss = loss_fn(output[mask], target[mask])
            losses.append(loss.item())

        loop.set_postfix(loss=loss.item())

    loop.set_postfix(loss=np.mean(losses))
    return np.mean(losses)

def train(model, train_loader, valid_loader, epochs=20, learning_rate=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        print(f"Epoch: {epoch+1}, Valid Loss: {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()
            torch.save(best_model, "models/best_model.pth")

    print("Training complete.")