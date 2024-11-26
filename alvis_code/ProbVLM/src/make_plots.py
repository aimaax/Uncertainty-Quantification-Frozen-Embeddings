import numpy as np
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


def main():
    metrics = np.load("../ckpt/ProbVLM_metrics.npy", allow_pickle=True).item()
    save_path = "../ckpt/"
    model_name = "ProbVLM"
    plot_loss(metrics, save_path, model_name)
    plot_MSE(metrics, save_path, model_name)
    plot_MAE(metrics, save_path, model_name)
    plot_MSE_MAE(metrics, save_path, model_name)


if __name__ == "__main__":
    main()


