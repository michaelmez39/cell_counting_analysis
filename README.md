# Automatic Cell Counting
Utilities to help count cells in hydrogels

## Sources
### Convolutional Neural Networks
**Countception**
Convolutional neural network that counts cells using a smaller CNN window to the image.\
[repo](https://github.com/ieee8023/countception)\
[paper](https://arxiv.org/abs/1703.08710)

**U-Net**
Also a CNN, considers whole image as input, uses pooling in network. Can also be used for image segmentation, which we might consider if we have time. Higher training time?
[easy explanation](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)
[paper segmentation](https://arxiv.org/abs/1505.04597)
[paper counting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8153189/#R24)

## Datasets
**[Modified Bone Marrow (MBM) Dataset](https://github.com/ieee8023/countception/blob/master/MBM_data.zip)**

**[VGG Cells Dataset](https://github.com/ieee8023/countception/blob/master/cells.zip)**
