import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.stats import entropy

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_false")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu
plot = True

update_interval = update_steps * batch_size

device = "cpu"

torch.manual_seed(seed)


torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Selecting classes 1,2,3,5,6,8
# idx = (dataset.targets==1) | (dataset.targets==2) | (dataset.targets==3) | (dataset.targets==5) | (dataset.targets==6) | (dataset.targets==8)  
# dataset.targets = dataset.targets[idx]
# dataset.data = dataset.data[idx]

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training.\n")
start = t()

model_to_save = None
running_perf = 0
confusion_matrix = np.zeros((10,10))
mae_from_uniform = []

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("\n Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    pbar_training = tqdm(total=n_train)
    for step, batch in enumerate(train_dataloader):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.

            # new_running_perf = 100 * torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor)
            # if (step > 50 and new_running_perf / running_perf > 1.8):
            #     question = "before: "+str(running_perf)+"after: "+str(new_running_perf)+"step: "+str(step) + " -- would you like to save (Y/N)?"
            #     save = input(question)
            #     if save == "Y":
            #         torch.save(network, "../saved models/"+str(n_neurons)+"_"+str(step*batch_size)+"_after")
            #         quit()
            
            # torch.save(network, "../saved models/"+str(n_neurons)+"_before")
            # running_perf = new_running_perf

            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )

            confusion_matrix = np.zeros((10,10))
            # Keep track of the confusion matrix
            for i,label_ in enumerate(label_tensor):
                real = label_tensor[i]
                pred = all_activity_pred[i]
                confusion_matrix[real][pred] += 1

            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.extend(batch["label"].tolist())

        input_exc_weights = network.connections[("X", "Ae")].w

        # Getting the weights before changing them for the sake of seeing the weight changes
        pre_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        input_exc_weights = network.connections[("X", "Ae")].w


        # Getting the weights after changing them for the sake of seeing the weight changes
        post_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )

        # The change of the weights from one batch
        weight_changes = post_weights - pre_weights
        weight_change_count = np.count_nonzero(weight_changes)
        


        # weight_change_count = np.count_nonzero(weight_changes)

        # change_arr.append(weight_change_count)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Optionally plot various simulation information.
        if step % update_steps == 0 and step > 0:
            if plot:
                image = batch["image"][:, 0].view(28, 28)
                inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
                lable = batch["label"][0]
                input_exc_weights = network.connections[("X", "Ae")].w
                square_weights = get_square_weights(
                    input_exc_weights.view(784, n_neurons), n_sqrt, 28
                )
                # weights_im = plot_weights(square_weights, im=weights_im, save="../weights/"+str(step)+".png")
                # perf_ax = plot_performance(
                #     accuracy, x_scale=update_steps * batch_size, ax=perf_ax
                # )


                # weight_changes = torch.from_numpy(normalize(weight_changes))
                # weight_changes = get_square_weights(weight_changes.view(784, n_neurons), n_sqrt, 28)
                # save_loc = "../weight_changes/"+str(step)+".png"
                # weights_im = plot_weights(weight_changes, im=weights_im, save=save_loc)


                fig, ax = plt.subplots()
                im = ax.imshow(confusion_matrix)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                for i in range(10):
                    for j in range(10):
                        text = ax.text(j, i, confusion_matrix[i, j],
                                    ha="center", va="center", color="w")

                ax.set_title("Confusion matrix of MNIST w/ SNN at " + str(step * batch_size))
                ax.set_xlabel("predicted label")
                ax.set_ylabel("true label")
                fig.tight_layout()
                plt.savefig("../confusion_matrices/"+str(step)+".png")
                plt.close(fig)

                fig, ax = plt.subplots()
                prediction_freq = np.sum(confusion_matrix, axis=0)
                ax.bar(np.arange(10), prediction_freq)
                ax.set_xlabel("predicted label")
                ax.set_ylabel("number of predictions")
                plt.savefig("../confusion_matrices/freq_"+str(step)+".png")
                plt.close(fig)

                fig, ax = plt.subplots()
                mae_from_uniform_current = (entropy(prediction_freq) * 100) / 2.3
                mae_from_uniform.append(mae_from_uniform_current)
                ax.plot(np.arange(len(accuracy["all"])), accuracy["all"], label="accuracy")
                ax.plot(np.arange(len(accuracy["all"])), mae_from_uniform, label="entropy")
                ax.set_title("correlation between entropy and accuracy")
                ax.set_xlabel("number of batches seen")
                ax.set_ylabel("accuracy (percent)")
                ax.legend()
                plt.savefig("../entropy_vs_acc/"+str(step)+".png")
                plt.close(fig)

                print(np.corrcoef(mae_from_uniform, accuracy["all"]))

                plt.pause(1e-8)

        # indices = np.arange(len(change_arr) )
        # plt.plot(indices, change_arr, color='r')
        # plt.xlabel('training batch number')
        # plt.ylabel('number of weights changed')
        # plt.show()
        # plt.title("Number of weights changed per update (400 neruons)")
        # plt.pause(1e-8)


        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

plt.show()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

