import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """
    Parses a log file to extract epoch, global_step, and average loss.

    Args:
        file_path (str): The path to the log file.

    Returns:
        pd.DataFrame: A DataFrame with columns 'epoch', 'step', and 'loss'.
    """
    data = []
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find all relevant lines, including those where the loss value is on the next line.
    pattern = re.compile(r"Epoch (\d+), global_step (\d+), average loss: \n?([\d.]+)")
    matches = pattern.findall(content)

    for match in matches:
        epoch, step, loss = match
        data.append({
            "epoch": int(epoch),
            "step": int(step),
            "loss": float(loss)
        })

    return pd.DataFrame(data)

def plot_loss(df):
    """
    Plots and saves loss vs. step and loss vs. epoch.

    Args:
        df (pd.DataFrame): DataFrame containing the training data.
    """
    # Plot Loss vs. Step
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["loss"])
    plt.xlabel("Global Step")
    plt.ylabel("Average Loss")
    plt.title("Training Loss vs. Global Step")
    plt.grid(True)
    plt.savefig("loss_vs_step.png")
    plt.close()

    # Plot Average Loss vs. Epoch
    epoch_loss = df.groupby("epoch")["loss"].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_loss["epoch"], epoch_loss["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Average Training Loss vs. Epoch")
    plt.grid(True)
    plt.savefig("loss_vs_epoch.png")
    plt.close()

# Main execution
log_file = 'epochs_100_lr_0.0001_rank_32_ts_1755590201.391309_console.log'
loss_data = parse_log_file(log_file)
# loss_data.to_csv("loss_data.csv", index=False)

plot_loss(loss_data)

print("Plots have been saved as loss_vs_step.png and loss_vs_epoch.png")
print("Extracted data has been saved to loss_data.csv")