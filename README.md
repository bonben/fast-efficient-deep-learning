# Fast track: Efficient Deep Learning

## 1. Train a ResNet-18 model on CIFAR-10 dataset.

Install necessary libraries:
```bash
pip install torchinfo wandb onnx onnx-simplifier
```

Create a Weights & Biases account to track your experiments.
Go to https://wandb.ai/site and sign up.
Then login via terminal:
```bash
export PATH=$PATH:/net/npers/YOUR_USERNAME/.local/bin
wandb login
```
Your API key can be found in your W&B account settings.

Your PATH variable should include the path to your local bin directory. You can add it to your `~/.bashrc` file:
```bash
export PATH=$PATH:/net/npers/YOUR_USERNAME/.local/bin
```

Launch the training script:
```bash
python main.py
```

## 2. Introductory course on Deep Learning

The folder `efficient-deep-learning` contains all of the materials for the full course "Efficient Deep Learning".
We are going to go through only a subset of the materials during this fast track.
Starting with `intro.pdf`, which contains an overview of the course and `course1.pdf`, which covers the basics of deep learning.

The content of the course is very close to the one presented earlier in https://gbourmaud.github.io/files/intro_deep_learning/cours/Cours_2025_2026_DL_1.pdf

We will see again with a new presentation, and will complete with a quizz destined to see neural networks under a different angle, their complexity: what are the number of parameters and the number of operations involved in each layer of a neural network during inference.

Join the wooclap for this course: https://app.wooclap.com/join/EDLENSEIRB

## 3. Modifying the ResNet-18 model

You will modify the files (`resnet.py`, `main.py`) to implement a new ResNet-12 model.
The ResNet-12 model should have the following architecture:

<-- will complete later -->

The reference implementation is readable as an onnx file: `resnet12-ref.onnx`.

The provided `main.py` also export the trained model as an onnx file: `trained-model.onnx`.

You can visualize the onnx files using Netron: https://netron.app

You should aim to have the same architecture in the exported onnx (`trained-model.onnx`) as in the reference onnx (`resnet12-ref.onnx`).

## 4. Course 2: Data Augmentation and Regularization

The document `efficient-deep-learning/course2.pdf` contains the materials for the second course on Data Augmentation and Regularization.

## 5. Application: Cutmix

In the `main.py`, Mixup is already implemented.
You will implement Cutmix as an additional data augmentation technique.
Add it using the `v2.CutMix` class from the `torch-vision-extensions` library.
Visualize the input images after applying Cutmix to verify your implementation and compare the results with Mixup.

## 6. Course 3: Pruning
The document `efficient-deep-learning/course3.pdf` contains the materials for the third course on Pruning.

## 7. Application: Pruning our ResNets
Refer to the `efficient-deep-learning/lab3.md` for the lab session on pruning and quantization.
Update the parameter count extracted from torchinfo.summary in `main.py` before and after pruning to verify the reduction in model size.

## 9 Course 4: Factorization

The document `efficient-deep-learning/course4.pdf` contains the materials for the fourth course on Factorization.

## 10. Application: Depthwise Separable Convolutions and Grouped Convolutions

Modify the ResNet-12 architecture to use Depthwise Separable Convolutions or Grouped Convolutions instead of standard convolutions.
Compare the performance and parameter count of the modified model with the original ResNet-12 model.

