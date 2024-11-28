# 99.4 % Accuracy for MNIST with just 14k parameter CNN and 20 epochs

[![Model Architecture Tests](https://github.com/nishantb06/mnist-drilldown/actions/workflows/model-checklist.yaml/badge.svg)](https://github.com/nishantb06/mnist-drilldown/actions/workflows/model-checklist.yaml)

## Logs

Model size for the NetV3 model:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.05
Estimated Total Size (MB): 1.12
----------------------------------------------------------------
```

Logs for the NetV3 model:

```
loss=0.21186162531375885 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.19it/s]

Epoch: 1
Test set: Average loss: 0.0779, Accuracy: 9839/10000 (98.39%)

loss=0.07612607628107071 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.83it/s]

Epoch: 2
Test set: Average loss: 0.0431, Accuracy: 9887/10000 (98.87%)

loss=0.027269145473837852 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.76it/s]

Epoch: 3
Test set: Average loss: 0.0373, Accuracy: 9896/10000 (98.96%)

loss=0.07748689502477646 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.10it/s]

Epoch: 4
Test set: Average loss: 0.0302, Accuracy: 9903/10000 (99.03%)

loss=0.05933946371078491 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.63it/s]

Epoch: 5
Test set: Average loss: 0.0396, Accuracy: 9882/10000 (98.82%)

loss=0.053781211376190186 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 18.98it/s]

Epoch: 6
Test set: Average loss: 0.0229, Accuracy: 9929/10000 (99.29%)

loss=0.05578482151031494 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.93it/s]

Epoch: 7
Test set: Average loss: 0.0243, Accuracy: 9922/10000 (99.22%)

loss=0.01884542964398861 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.11it/s]

Epoch: 8
Test set: Average loss: 0.0234, Accuracy: 9932/10000 (99.32%)

loss=0.04304708167910576 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.91it/s]

Epoch: 9
Test set: Average loss: 0.0289, Accuracy: 9910/10000 (99.10%)

loss=0.023164687678217888 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.76it/s]

Epoch: 10
Test set: Average loss: 0.0228, Accuracy: 9926/10000 (99.26%)

loss=0.04047480598092079 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.55it/s]

Epoch: 11
Test set: Average loss: 0.0233, Accuracy: 9913/10000 (99.13%)

loss=0.07659625262022018 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.76it/s]

Epoch: 12
Test set: Average loss: 0.0217, Accuracy: 9925/10000 (99.25%)

loss=0.006894339341670275 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.44it/s]

Epoch: 13
Test set: Average loss: 0.0238, Accuracy: 9928/10000 (99.28%)

loss=0.030943594872951508 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.50it/s]

Epoch: 14
Test set: Average loss: 0.0238, Accuracy: 9926/10000 (99.26%)

loss=0.03204219788312912 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.95it/s]

Epoch: 15
Test set: Average loss: 0.0229, Accuracy: 9928/10000 (99.28%)

loss=0.04895944893360138 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.75it/s]

Epoch: 16
Test set: Average loss: 0.0193, Accuracy: 9932/10000 (99.32%)

loss=0.02402343787252903 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.67it/s]

Epoch: 17
Test set: Average loss: 0.0209, Accuracy: 9930/10000 (99.30%)

loss=0.0037274854257702827 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.14it/s]

Epoch: 18
Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

loss=0.019095933064818382 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 20.40it/s]

Epoch: 19
Test set: Average loss: 0.0184, Accuracy: 9941/10000 (99.41%)

loss=0.002912658965215087 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.24it/s]

Epoch: 20
Test set: Average loss: 0.0184, Accuracy: 9936/10000 (99.36%)
```
