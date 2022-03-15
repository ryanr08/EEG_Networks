
# Evaluation of Neural Networks for Classifying Electroencephalography (EEG) Data

  

#### Rui Zheng, Ryan Riahi, Mao Kai Yu, Bangyao Hu

  

## Abstract

In this project, we wish to accurately classify EEG brain data in order to predict human activity. There are many useful applications for understanding human behavior through brain scans, and thus we wish to obtain accurate predictions. Convolutional neural networks (CNN), recurrent neural networks (RNN), and generative adversarial neural networks (GAN) are popular and powerful tools used in the field of deep learning. In our project, we used these three neural network architectures to attempt to best classify EEG data sampled from four human tasks. We evaluated these networks on their own, as well as combining some of them. From our analysis, we learned that CNNs appear to be best for classifying EEG data, while using an RNN performed fairly poorly. Combining the CNNs and the RNNs did not yield great results, as only using the plain CNN had better accuracy. Lastly, the GAN-generated data added some improvement to our classification accuracy, especially when there was not adequate data for training.

  

## Introduction
    

EEG data contains information about millions of neuron's activity within the brain. 
The dataset can be found at: https://www.bbci.de/competition/iv/. More information on the dataset is located at: https://www.bbci.de/competition/iv/desc_2a.pdf.

The data was taken via electrodes connecting to the scalp of the head and recorded four seperate tasks: Movement of the left hand, right hand, both feet, and tonque.
Therefore, our main goal is to accurately classifying the EEG into one of the four classes.  A variety of methods were used in classifying EEG data, such as support vector machine, hidden Markov models, k-nearest neighbors, etc. [4] Since the signal-to-noise ratio of EEG data is usually very low, many preprocessing techniques were required for most of these methods. However, these techniques might have adverse effects on the classification results while NN requires little or no preprocessing procedures. [2]

  
In this project, Several NN models were compared. We first start tackling the EEG classification problem with CNN to serve as a baseline of the whole project. After that, RNN including LSTM and GRU were implemented with the intuition that they would perform better on temporal problems. We also attempted various combinations of CNNs with LSTMs and GRUs. We then attempted to enlarge the dataset using GANs including WGAN and ACGAN to generate more data. We will compare the benefits of adding in the generated data as opposed to just training with the baseline data. In addition, these models were run not only on individual subject but also across all 9 datasets to observe the effect of individual differences on EEG data.

 ### CNN
    

CNN was the first architecture we started with, and the one we ended up having the most success with. The first reason we implemented CNN at the beginning is that CNN can serve as a good baseline for our project since it is widely used for various kinds of classification problems. The second reason is that it has the ability to easily withdraw features from data with few pre-processing procedures. The third reason is that CNN is commonly used as an excellent image classifier and we think that each EEG sample can be viewed as a one-dimensional image consisting of 22 color channels of voltage pixels along a time scale as well.

  

We first built a shallow CNN with 3 convolution layers and 1 pooling layer. The kernel size is (1, 1) with the intuition that a smaller kernel size usually performs better in CNN classifiers. However, it didn’t perform well. Noticing that the time step dimension is much larger than the voltage dimension, and the kernel size of (1, 1) might be too small to obtain features across different time steps, we built a deeper CNN with a larger kernel size (10, 1) in time step dimensions. This time, 4 convolution layers were used, each following a pooling layer, dropout layer, and batch normalization layer to avoid overfitting. Each convolutional layer included some padding (5, 0) to maintain the output size as well as a stride of (1, 1). We convoluted across time, with each filter having a depth of 22, one for each channel of the EEG data. At the end of the CNN, we only had one affine layer followed by a softmax classifier. Lastly, we used the adam optimizer when training over 100 epochs. The result of the deeper CNN model turns out to be pretty well.

 ### RNN
    

With the introduction of internal state units, RNNs usually perform well when processing sequential input such as temporal dynamic data. Considering EEG data’s time-related nature, we then implemented RNNs to see whether they could improve the classification performance. To avoid the vanishing and exploding gradients problem and gain better performance, both LSTM and GRU are investigated in this project. We try to use LSTM and GRU layers to withdraw features from the time dimension of raw data at the beginning. But their testing accuracies were much lower than the above CNN. Assuming large scale on time-dimension impeded the training of LSTM layers and GRU layers, we implemented CNN+LSTM and CNN+GRU models to reduce the length of time-dimension before introducing RNN layers. To easily compare the effect of introducing RNN layers, the CNN part used in CNN+LSTM and CNN+GRU models are the same as the deeper CNN introduced above.

  

### LSTM

For the LSTM, we built three LSTM layers stacked together with dropout. This was then followed by several FC layers with BN and ReLu activation.

### GRU

For the GRU, we also use 3 GRU layers stacked together with dropout, in comparison with LSTM. And it was also followed by several FC layers with BN and ReLu activation.

### GAN
    

When using the above models to classify the EEG data, we find that the test accuracy is much lower than the training accuracy, which indicates a significant overfitting problem. It is known that increasing the sample size is an effective way to solve overfitting. GAN[101] generates fake data through a generator network and discriminates the data from true and fake through a discriminator network. In the process of finding the Nash Equilibrium Point of these two networks, the generator fits the distribution of the true data. After the training phase, the generator is used to generate fake samples similar to the true data, and we want to perform data augmentation using fake samples to deal with the overfitting problem

### WGAN
    

If the discriminators in the GAN are trained to be optimal, then the goal of the whole GAN is to minimize the JS divergence between the real data and the fake data. However, the bad nature of JS divergence leads to gradient vanish. Specifically, if the discriminator is too good, the generator cannot get enough gradients for optimization, yet if the discriminator is too poor, it cannot guide the generator to learn effectively. This is the reason why GAN is difficult to train. Martin Arjovsky[102] proposed the Wasserstein distance is a more suitable measure of distribution difference than JS divergence, and its gradient has better properties. Ishaan Gulrahani[103] adds a gradient penalty to the loss function of the discriminator, so that the discriminator can estimate the Wasserstein distance and use it to guide the generator, obtaining a more stable training result and mitigating the mode collapse problem to some extent. In this project, we implement WGAN to generate fake data to improve the downstream classification task. The discriminator takes EEG data and their labels as inputs, extracts features by multi-layer convolution, and estimates the Wasserstein distance by the final fully connected layer. Meanwhile, the generator generates fake EEG data using multi-layer transposed convolution.

### ACGAN
    

Although WGAN avoids the gradient vanish and mode collapse problems to some extent, WGAN only takes the labels of data as input of the generator and discriminator, and in the process of training the network to fit the distribution of EEG data, it is difficult for the network to discover the connection between EEG data and its labels, which is detrimental to the downstream EEG classification task. ACGAN[104] takes the labels of data only as input of the generator, requiring the discriminator to predict the labels of the data while separating the true from the false data, and incorporating the classification loss into the loss function. In this project, we modify the structure of the discriminator based on WGAN’s architecture and network parameters, and incorporate the classification loss to its loss function. We hope to use the idea of ACGAN to make the network capture the connection between data and its label while learning the distribution of EEG data, in order to better assist the downstream classification task.

## Results
    

![](https://lh4.googleusercontent.com/6vnhD2-2n9C45U_qw8G0Q0wqE4WUQDylaJw1B3FFYDKP5hluIQQUcp6JOqoBDpl0Fg2tc3xRVxkLYR2CSG92xmO6_9qVgQJFuofj33K7QS2a-HhThB5ySh7fZZsm9Q)

  

![](https://lh4.googleusercontent.com/_phUlEvP9MhWbf8zsrmN3M4WeGy4773PDeEwaKjtRZPT-Z36ZvObpBN7PEn5D7O-nG__gjFPQF7FZakD5OskP94GRid6JIp7nxyF27-2rNmPEQQ-ovCVF9fJqB32Gg)

## Discussion
    

### Single subject VS all subjects
    

For the accuracy of our classifier, training a given network for a single subject rather than generalizing for all subjects improved performance. For every single network we trained, the testing accuracy increased by about 0.05 - 0.1%. This makes some intuitive sense since having a network personally tailored to each subject would be able to sort of “overfit” to that subject.

At the same time, the networks trained on all the subjects also did relatively well, the accuracies are only slightly behind for a single subject. This also makes sense. We are given more training data when we train on all subjects. Additionally, having to generalize across subjects could be beneficial in certain ways when predicting for a single subject. Therefore, while specializing in a single subject improves accuracy, classifying all subjects still works very well.

### CNN
    

Two different approaches were tried for CNN. The first was a shallow CNN with more fully connected (FC) layers, which performed worse than the deeper CNN with fewer FC layers. Both CNNs initial convolute across time, but the shallow CNN only has 3 convolutional layers and 1 pooling layer while the deep CNN has 4 convolution layers and 4 pooling layers. The shallow CNN has 4 fully connected layers, while the deep CNN has only 1. Overall, the deep CNN (73% accuracy) far outperformed the shallow one (33% accuracy).

Out of all the neural networks, CNN ended up having the best accuracy of them all. Across all subjects with original data and GAN added data, the CNN had the best test accuracy with 65% and 70% respectively. The highest accuracy we achieved was 73%, from the CNN with the original data across one subject. The best the RNNs and CNNs + RNNs could do for all subjects was 63% accuracy, and for one subject was 71% (CNN + GRU). We are unsure of the reasoning for the CNN performing the best. One possible reason is that convoluting across the time with the EEG data works well since the data around a particular action (i.e. raising of the hand, …) has a certain temporal relationship well suited to convolution (similar to how image data’s spatial relationship is well suited for convolution). Another possibility is that EEG data has a very little temporal relationship, and thus CNNs perform better than RNNs.

### LSTM & GRU
    

The LSTM and GRU did not perform well with the EEG data. When on their own, their accuracy was between 30-40% for all subjects and one subject. They were far outperformed by CNN, which has proven to be the best model for EEG that we tested. There is always the possibility that our design of the RNNs was not optimal, and certain hyperparameters need to be altered / layers need to be redesigned. One theory is that the EEG data is not temporally related in the way that a video or text is. With text, the content hundreds of characters ago could be relevant presently, but with EEG data we could be seeing that it’s more similar to areas of time resembling a particular action.

  

### CNN combined with RNN
    

Combining the CNN and RNN did not seem to do much. While it performed better than the plain RNNs, adding the RNNs only seemed to hurt the already existing CNN. One potential fix that would have tried is to remove the dropout layers in the CNN and add one after the LSTM layers. This could have helped avoid the LSTM layers missing too much information.

  
  

### GAN generated VS original data
    

For the case of considering subject 1, we added 25% of the fake data generated by ACGAN to the original samples. The results show that GAN_based data augmentation reduces the accuracy of CNN, CNN+GRU, and weakly increases the accuracy of GRU, CNN+LSTM. We consider that only using data from subject 1, the amount of true samples is too small to support GAN training and leads to poor quality of the generated samples.

For the case considering all subjects, the data augmentation method improves the accuracy of all models except the LSTM, and ACGAN is better than WGAN. We consider that in this case, the amount of true samples is sufficient to support the GANs to generate suitable fake data and to help the training of downstream neural networks. And after incorporating the classification loss into the discriminator loss, the neural network is better at discovering the connection between data and its labels in order to generate category-relevant samples.

To summarize, we found that when the amount of true data is too small, the GAN is not able to generate high-quality samples. When the amount of true data is very large, the classification network does not need GAN-based data augmentation to prevent overfitting. However, when the amount of true data is medium, GAN can generate high-quality samples to expand the dataset and assist the training of downstream networks. In addition, considering the loss function of the downstream task in advance during the training of GANs can generate more specific samples in order to better improve the performance of the downstream task.

## References
    


[4] P. Wang, A. Jiang, X. Liu, J. Shang, and L. Zhang. Lstmbased

eeg classification in motor imagery tasks. IEEE Transactions

on Neural Systems and Rehabilitation Engineering,

26(11):2086–2095, 2018.

  

[2] D. Kostas, E. W. Pang, and F. Rudzicz. Machine learning for

meg during speech tasks. Scientific reports, 9(1):1–13, 2019

  

[100] Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

[101] Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein gan. arXiv 2017." arXiv preprint arXiv:1701.07875 30 (2017): 4.

[102] Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C. Courville. "Improved training of wasserstein gans." Advances in neural information processing systems 30 (2017).

[103] Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." In International conference on machine learning, pp. 2642-2651. PMLR, 2017.  

the performance of all algorithms

![](https://lh4.googleusercontent.com/6vnhD2-2n9C45U_qw8G0Q0wqE4WUQDylaJw1B3FFYDKP5hluIQQUcp6JOqoBDpl0Fg2tc3xRVxkLYR2CSG92xmO6_9qVgQJFuofj33K7QS2a-HhThB5ySh7fZZsm9Q)

Figure . Accuracy for subject1

  

![](https://lh4.googleusercontent.com/_phUlEvP9MhWbf8zsrmN3M4WeGy4773PDeEwaKjtRZPT-Z36ZvObpBN7PEn5D7O-nG__gjFPQF7FZakD5OskP94GRid6JIp7nxyF27-2rNmPEQQ-ovCVF9fJqB32Gg)

  

Figure . Accuracy for all subjects

  
  

![](https://lh6.googleusercontent.com/_y6wa2Jvtu3nXKqFrnLfBP9IbkhqjRtpB11d-KwtHGDzRKfX7L9weW-hj6EZzY_ynX68AAlwAdVsey8Rx0rgtN_5U1ArdO109mjVJWSYime67O5D9S2O4hzY5oVaNg)

Figure . Loss Curve of WGAN for all subjects![](https://lh5.googleusercontent.com/y9KrwiTYhzHi-855QIcOCouu5HoXFE6RAA6p25PE78QOUwKSOz7kv4fwBI21rTKBX2wYUQc0RUHJ7Bbd_NeLvAHG5cRe6J2jAJKmRiwbJ7RXMkUfjQ7eK8LCoceqTQ)

Figure . Loss Curve of ACGAN for all subjects

  

![](https://lh5.googleusercontent.com/7x7ZU23OP57Jo5lWZmWcVJxQz4QY_7Q-1fwjepeF50dqmcumT4uYAg3DMtXXTH6-cfQBnxmH_5h2Pl5txOzkGhWMX3eTxPGKDaT3vEv2n9oRPBIqT21-HLkQwSZmhg)

Figure . A sample generated by WGAN (17th channel, label_1)

  

![](https://lh4.googleusercontent.com/K30WMNUhMuk1uRWLleYIWRExBapWwiLzuKpGCCQr1J19OpLsb5OF2xhqNp7-xVVHO8-SUXHmyVm9huUMoOOOn9qO91WxwQBLPwAFi_7iLqbvlC8f3U5eNd6Dz8A6rA)

Figure . A sample generated by ACGAN(17th channel, label_1)

 
