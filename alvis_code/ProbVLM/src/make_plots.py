import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_loss(metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epochs"], metrics["train_loss"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_loss.png" 
    plt.savefig(path, dpi=300)  


def plot_MAE(metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epochs"], metrics["val_mae"], label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Validation MAE vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MAE.png" 
    plt.savefig(path, dpi=300)


def plot_MSE(metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epochs"], metrics["val_mse"], label="Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Validation MSE vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MSE.png" 
    plt.savefig(path, dpi=300)


def plot_MSE_MAE(metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["epochs"], metrics["val_mse"], label="Validation MSE")
    plt.plot(metrics["epochs"], metrics["val_mae"], label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Validation Metrics vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MSE_MAE.png" 
    plt.savefig(path, dpi=300)

def convert_tensors_in_lists(data):
    """
    Recursively convert tensors in lists or nested structures to NumPy arrays.
    """
    if isinstance(data, list):
        return [convert_tensors_in_lists(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return data

def main():
    # Load the metrics file
    #metrics = np.load("../ckpt/BBB_EncBL_metrics.npy", allow_pickle=True).item()
    metrics = np.load("../ckpt/ProbVLM_Net_metrics.npy", allow_pickle=True).item()
    # Convert any tensors within lists or values to NumPy arrays
    for key in metrics:
        metrics[key] = convert_tensors_in_lists(metrics[key])

    save_path = "../figs/"
    #model_name = "BBB_EncBL"
    model_name = "ProbVLM"
    plot_loss(metrics, save_path, model_name)
    plot_MSE(metrics, save_path, model_name)
    plot_MAE(metrics, save_path, model_name)
    plot_MSE_MAE(metrics, save_path, model_name)

    print(f"Plots saved to {save_path}")


if __name__ == "__main__":
    main()


