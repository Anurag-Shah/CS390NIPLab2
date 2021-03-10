import matplotlib.pyplot as plt

datasets = ["MNIST\nDigits", "MNIST\nFashion", "CIFAR 10", "CIFAR 100\nCoarse", "CIFAR 100\nFine"]
ann_accuracy = [98.07, 89.11, 38.3, 22.93, 11.99]
cnn_accuracy = [99.22, 92.99, 74.52, 54.06, 41.22]

plt.bar(datasets, ann_accuracy)
plt.title("ANN Accuracy Plot by Dataset")
plt.xlabel("Datasets")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("ANN_Accuracy_Plot.pdf")

plt.bar(datasets, cnn_accuracy)
plt.title("CNN Accuracy Plot by Dataset")
plt.xlabel("Datasets")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("CNN_Accuracy_Plot.pdf")