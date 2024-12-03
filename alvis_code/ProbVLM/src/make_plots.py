import numpy as np
import re
import torch
import matplotlib.pyplot as plt

def ensure_numpy_array(data):
    """
    Ensure the data is converted to a NumPy array.
    Handles PyTorch tensors (both CPU and CUDA), lists, and already existing NumPy arrays.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()  # Move to CPU and convert to NumPy
    elif isinstance(data, list):
        return np.array([ensure_numpy_array(item) for item in data])  # Recursively handle nested lists
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)  # Convert other types (e.g., plain numbers) to NumPy arrays


def plot_loss(metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), ensure_numpy_array(metrics["train_loss"]), label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_loss.png"
    plt.savefig(path, dpi=300)


def plot_MAE(metrics, save_path="../ckpt/", model_name="model"):
    val_mae = ensure_numpy_array(metrics["val_mae"])
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), val_mae, label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Validation MAE vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MAE.png"
    plt.savefig(path, dpi=300)


def plot_MSE(metrics, save_path="../ckpt/", model_name="model"):
    val_mse = ensure_numpy_array(metrics["val_mse"])
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), val_mse, label="Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Validation MSE vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MSE.png"
    plt.savefig(path, dpi=300)


def plot_MSE_MAE(metrics, save_path="../ckpt/", model_name="model"):
    val_mae = ensure_numpy_array(metrics["val_mae"])
    val_mse = ensure_numpy_array(metrics["val_mse"])
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), val_mse, label="Validation MSE")
    plt.plot(ensure_numpy_array(metrics["epochs"]), val_mae, label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Validation Metrics vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_MSE_MAE.png"
    plt.savefig(path, dpi=300)


def plot_kl(metrics, save_path="../ckpt/", model_name="model"):
    kl = ensure_numpy_array(metrics["kl"])
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), kl, label="KL Divergence Term")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence Term")
    plt.title("Loss KL Divergence Term vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_KL.png"
    plt.savefig(path, dpi=300)


def extract_kl_values_from_slurm(file_path):
    kl_values = []

    # Regular expression to match the KL tensor values
    kl_pattern = r"kl: tensor\(\[([\d\.\-e]+)\], device='cuda:\d+'"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(kl_pattern, line)
            if match:
                # Extract the numeric value and convert to float
                kl_value = float(match.group(1))
                kl_values.append(kl_value)

    return kl_values


def plot_extracted_kl(kl_values, metrics, save_path="../ckpt/", model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(ensure_numpy_array(metrics["epochs"]), ensure_numpy_array(kl_values), label="KL Divergence Term")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence Term")
    plt.title("Loss KL Divergence Term vs. Epochs")
    plt.legend()
    plt.grid()
    path = save_path + model_name + "_KL.png"
    plt.savefig(path, dpi=300)


def plot_parameters(parameters, save_path="../ckpt/", model_name="model"):
    txt_mu_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                               for val in parameters["txt_mu"]], dim=0)
    txt_alpha_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                                  for val in parameters["txt_alpha"]], dim=0)
    txt_beta_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                                 for val in parameters["txt_beta"]], dim=0)
    img_mu_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                               for val in parameters["img_mu"]], dim=0)
    img_alpha_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                                  for val in parameters["img_alpha"]], dim=0)
    img_beta_vals = torch.stack([torch.tensor(val) if not isinstance(val, torch.Tensor) else val 
                                 for val in parameters["img_beta"]], dim=0)

    # Create the epoch numbers (x-axis)
    epochs = np.arange(txt_mu_vals.shape[0])

    # Plot txt_mu
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, ensure_numpy_array(txt_mu_vals.mean(dim=(1, 2))))
    plt.title("Text Mu")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")

    # Plot txt_alpha
    alpha_vals_txt = 1 / (1e-2 + ensure_numpy_array(txt_alpha_vals.mean(dim=(1, 2))))
    plt.subplot(2, 3, 2)
    plt.plot(epochs, alpha_vals_txt)
    plt.title("Text Alpha")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Alpha Value")

    # Plot txt_beta
    plt.subplot(2, 3, 3)
    plt.plot(epochs, ensure_numpy_array(txt_beta_vals.mean(dim=(1, 2))))
    plt.title("Text Beta")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Beta Value")

    # Plot img_mu
    plt.subplot(2, 3, 4)
    plt.plot(epochs, ensure_numpy_array(img_mu_vals.mean(dim=(1, 2))))
    plt.title("Image Mu")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Value")

    # Plot img_alpha
    alpha_vals_img = 1 / (1e-2 + ensure_numpy_array(img_alpha_vals.mean(dim=(1, 2))))
    print(alpha_vals_img)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, alpha_vals_img)
    plt.title("Image Alpha")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Alpha Value")

    # Plot img_beta
    plt.subplot(2, 3, 6)
    plt.plot(epochs, ensure_numpy_array(img_beta_vals.mean(dim=(1, 2))))
    plt.title("Image Beta")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Beta Value")

    plt.subplots_adjust(hspace=0.4)

    path = save_path + model_name + "_parameters.png"
    plt.savefig(path, dpi=300)


def make_all_plots(metrics, parameters, save_path, model_name):
    # Ensure all tensors and lists are converted to NumPy arrays
    metrics = {key: ensure_numpy_array(value) for key, value in metrics.items()}
    parameters = {key: [ensure_numpy_array(param) for param in value] for key, value in parameters.items()}

    # Plot metrics
    plot_loss(metrics, save_path, model_name)
    plot_MSE(metrics, save_path, model_name)
    plot_MAE(metrics, save_path, model_name)
    plot_MSE_MAE(metrics, save_path, model_name)
    #plot_kl(metrics, save_path, model_name)
    print(f"Plots saved to {save_path}")
    # Plot parameters
    plot_parameters(parameters, save_path, model_name)

    
    print(f"Parameters plot saved to {save_path}")


def main():
    #metrics1 = np.load("../ckpt/BBB_EncBL_metrics_1.npy", allow_pickle=True).item()
    #parameters1 = np.load("../ckpt/BBB_EncBL_parameters_1.npy", allow_pickle=True).item()
    #save_path1 = "../figs/"
    #model_name1 = "BBB_EncBL_1"

    #metrics2 = np.load("../ckpt/BBB_EncBL_metrics_2.npy", allow_pickle=True).item()
    #parameters2 = np.load("../ckpt/BBB_EncBL_parameters_2.npy", allow_pickle=True).item()
    #save_path2 = "../figs/"
    #model_name2 = "BBB_EncBL_2"

    #metrics3 = np.load("../ckpt/BBB_EncBL_metrics_3.npy", allow_pickle=True).item()
    #parameters3 = np.load("../ckpt/BBB_EncBL_parameters_3.npy", allow_pickle=True).item()
    #save_path3 = "../figs/"
    #model_name3 = "BBB_EncBL_3"
    
    metrics3 = np.load("../ckpt/ProbVLM_Net_metrics_3.npy", allow_pickle=True).item()
    parameters3 = np.load("../ckpt/ProbVLM_Net_parameters_3.npy", allow_pickle=True).item()
    save_path3 = "../figs/"
    model_name3 = "ProvVLM_3"

    metrics2 = np.load("../ckpt/ProbVLM_Net_metrics_2.npy", allow_pickle=True).item()
    parameters2 = np.load("../ckpt/ProbVLM_Net_parameters_2.npy", allow_pickle=True).item()
    save_path2 = "../figs/"
    model_name2 = "ProvVLM_2"
    
    metrics1 = np.load("../ckpt/ProbVLM_Net_metrics_1.npy", allow_pickle=True).item()
    parameters1 = np.load("../ckpt/ProbVLM_Net_parameters_1.npy", allow_pickle=True).item()
    save_path1 = "../figs/"
    model_name1 = "ProvVLM_1"
    
    make_all_plots(metrics1, parameters1, save_path1, model_name1)
    make_all_plots(metrics2, parameters2, save_path2, model_name2)
    make_all_plots(metrics3, parameters3, save_path3, model_name3)


if __name__ == "__main__":
    main()


