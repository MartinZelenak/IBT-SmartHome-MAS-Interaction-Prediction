from typing import List

import matplotlib.pyplot as plt


def plot_losses( losses: List[float], file_path: str, title: str = "Losses", xlabel:str = "Step", ylabel: str = "Loss" ) -> None:
    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(losses)), losses, s=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path)
    plt.close()

def plot_average_loss( losses: List[float], window_size: int, file_path: str, title: str = "Average loss", xlabel: str = "Step", ylabel: str = "Loss", ) -> None:
    averaged_losses = [
        sum(losses[i : i + window_size]) / len(losses[i : i + window_size])
        for i in range(0, len(losses), window_size)
    ]
    plt.figure(figsize=(20, 6))
    plt.plot(range(len(averaged_losses)), averaged_losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path)
    plt.close()
