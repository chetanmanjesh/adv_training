# image_classification_adversarial_attack

This code will train a ResNet/linear classifier over cifar10/mnist. 
And it will test with original test data and adversarial test data.
After training, 100 sampled images will be saved for both test input and adversarial one.

[TODO] current accuracy of resnet over CIFAR10 is 86.490%, which is not lower than 94%.
[TODO] train the model with adversarial input.

Environment:
python 3.6
pytorch 1.01
matplotlib

Command example:

`python main.py --model=resnet --data=cifar10`
