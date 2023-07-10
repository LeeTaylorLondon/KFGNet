"""
Author: Lee Taylor

plot_training.py : parse and plot the loss of training over epochs
"""
import matplotlib.pyplot as plt


with open("training_06072023.txt") as f:
    lines = f.readlines()

data = []
for line in lines:
    if line.__contains__("Total Loss:"):
        data.append(line)

print(data)

# Parse average loss from each string
avg_losses = [float(s.split('Avg Loss: ')[1]) for s in data]

# Create a list of epoch numbers
epochs = list(range(len(avg_losses)))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_losses, marker='o')

plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')

plt.grid()
plt.show()