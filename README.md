# Real-time abnormal human's motion behavior detection 
In the project, a deep learning method and a GMM-based motion detection algorithm are developed for detecting abnormal behavior.
The trained models are evaluated using accuracy and confusion chart, and two user-friendly GUIs are implemented 
for offline and real-time monitoring and analysis.

## Description

In recent years, there has been a surge in the use of surveillance cameras in public places 
such as parks, airports, and shopping malls to enhance public safety and social security. 
Despite this, many public places still rely on traditional surveillance techniques such as offline 
video analysis or manual monitoring to identify unusual behaviour. Given the exponential 
growth in surveillance video data, these traditional methods have become increasingly complex 
and challenging, preventing security personnel from detecting unusual events promptly. In addition, abnormal events typically occur far less frequently than typical activities, 
resulting in a significant waste of manpower and time. There is therefore an urgent need to 
develop an intelligent surveillance system that can analyze human behavior in real time, 
automatically alert and preserve evidence related to abnormal behavior. In addition, motion 
detection can accurately locate the position of moving individuals in video, simplifying the 
process for users to identify areas of interest when reviewing recorded alarm scenes or 
observing live footage. This ultimately improves the efficiency and accuracy of abnormal 
activity detection

## Results 
### Classification 
![image](https://github.com/Andyli9926/Abnormal-Behavior-Dectection-in-Real-Time-and-Video/assets/145501579/375a75d8-dd21-498a-ad98-69de995bf847)
### StramingAPP
![image](https://github.com/Andyli9926/Abnormal-Behavior-Dectection-in-Real-Time-and-Video/assets/145501579/d715de2f-fd70-4901-b119-205ea84bb605)
### VideoAPP
![image](https://github.com/Andyli9926/Abnormal-Behavior-Dectection-in-Real-Time-and-Video/assets/145501579/52964d4e-9e8c-44be-a1ed-b3c96e2603b8)
## Getting Started

### Required Toolboxes

* Deep Learning Toolbox
* Computer Vision Toolbox
* Deep Learning Toolbox Model for GoogLeNet Network
* Matlab R2023b

### Installing

* Download the training codes in the train folder.
* If you want to use or modify the Apps, download the codes in the App folder or download the apps with the link https://drive.google.com/drive/folders/1nNl5e_9ZczLWtN3lFWkMv_V1-AiKhOX6?usp=sharing
* The model trained by myself can be downloaded in the link
https://drive.google.com/file/d/1_Y2Hnr-N38skxBRdqvdvuAhCjP7RChgs/view?usp=sharing

### Executing program
#### For training
* To run the train code, open the code in the live editor in MATLAB
* Change the file address to your dataset and change the names of each behavior type following the order of each folder.
* Change the address to store the trained model
#### For Apps
* To run Apps, change the model address to your model or my model.
* For the WebcamAPP, change the image store address to your address.
#### Others
* Run Classify codes in live editor
* Run Motiontracking code in editor


## Authors

Zekai Li

Email:zl874@cornell.edu


## Reference
* Carreira, Joao, and Andrew Zisserman. "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR): 6299??6308. Honolulu, HI: IEEE, 2017.
* Simonyan, Karen, and Andrew Zisserman. "Two-Stream Convolutional Networks for Action Recognition in Videos." Advances in Neural Information Processing Systems 27, Long Beach, CA: NIPS, 2017.
* Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." International Conferencee on Learning Representations 2017. Toulon, France: ICLR, 2017.
* Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri. "A Closer Look at Spatiotemporal Convolutions for Action Recognition". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6450-6459.
* Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. "SlowFast Networks for Video Recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
* Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, Andrew Zisserman. "The Kinetics Human Action Video Dataset." arXiv preprint arXiv:1705.06950, 2017.
