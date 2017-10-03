# Generative-Adversarial-Nets
Different GAN ([Generative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)) architectures in TensorFlow

## convGAN
A Generative Adversarial Net implemented with **TensorFlow** using the
**MNIST** data set.

#### Generator:
* Input: **100**
* Output: **784**
* Purpose: Will learn to **output images** that **look** like a **real**
image from **random input**. 



#### Discriminator:
* Input: **784**
* Output: **1**
* Purpose: Will learn to tell a **real** ("looks like it could be a real image in MNIST dataset") **image**(784) from a fake one.


#### Notes and Outputs
A problem with the way that I built this is that I used the **same architecture**
for **both** the **generator** and **discriminator**. Although I thought this save me, the developer, a lot of time it actually
caused a lot of problems with trying to pigeonhole that architecture to work with a smaller input **(Discriminator: 28x28 vs 10x10 : Generator)**. 

##### Architecture

* conv1 -> relu -> pool -> 
* conv2 -> relu -> pool ->
* conv3 -> relu -> pool ->
* fullyConnected1 -> relu ->
* fullyConnected2 -> relu -> 
* fullyConnected3 ->
![generated gan output](gan_generated.gif)
100 random numbers -> Generator -> ImageOutput -> Discriminator -> (Real|Fake) 


