# TDCN
Created by Mingchun Li & Dali Chen

### Introduction:

We propose an effective boundary detection network named TDCN based on transformer. 
Different with the pure transformer, it involves difference convolution when acquiring token 
embedding. The difference convolution including TAG layer explicitly extracts the gradient 
information closely related to boundary detection. 
Then these features are further transformed together with the dataset token through our 
transformer. Our boundary-aware attention in transformer and TAG layer achieve efficient 
feature extraction to keep the model lightweight. 
And the dataset token embedding gives our 
model the ability to universal predictions for multiple datasets. 
Finally, we use the bidirectional boosting strategy to train the head functions for 
multi-scale features. These strategies and designs ensure good performances of the model. 
And multiple experiments in this paper demonstrate the effectiveness of our method. 

### Prerequisites

- pytorch >= 1.7.1(Our code is based on the 1.7.1)
- numpy >= 1.11.0

### Train and Evaluation
1. Clone this repository to local

2. Download the datasets provided in [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf), and extract these datasets to the `$ROOT_DIR/data/` folder.
    ```
    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
    ```
3. Download the related .lst file for query. The link is https://drive.google.com/drive/folders/11k5Zqq-1fCxdl96CxoqPGKa87ssqazwx?usp=drive_link.

4. Run the training code main.py (for BSDS500) or main_multi.py (for NYUD, or mutiple datasets).

5. The metric code is in metric folder. It may require additional support libraries, please refer to [pdollar Repository](https://github.com/pdollar/edges).

We have released the final prediction and evaluation results, which can be downloaded at the following link:
https://pan.baidu.com/s/1z3G2N3PePY9ex1SG3wJXsw?pwd=7z7m Code：7z7m

### Final models
This is the final model in our paper. We used this model to evaluate. You can download by: 

https://pan.baidu.com/s/1m0Ufgl5rIcvi59yXKUedjg?pwd=65zn Code：65zn

### Acknowledgment
Part of our code comes from [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf), [Pidinet Repository](https://github.com/zhuoinoulu/pidinet). We are very grateful for these excellent works.
