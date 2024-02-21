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

### News !
We successfully apply the proposed attention loss to semantic segmentation tasks and achieve better performance 
than the original <a href='https://github.com/rstrudel/segmenter'>segmenter</a> without increasing any model parameters.

### ADE20K
Segmenter models with/without our attention loss:
<table>
  <tr>
    <th>Name</th>
    <th>mIoU(SS)</th>
    <th># params</th>
    <th>Resolution</th>
    <th colspan="3">Download</th>
  </tr>
<tr>
    <td>Seg-T-Mask/16</td>
    <td>38.1</td>
    <td>7M</td>
    <td>512x512</td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/variant.yml">config</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/log.txt">log</a></td>
  </tr>
<tr>
    <td>Seg-T-Mask/16(Our Reproduced)</td>
    <td>38.4</td>
    <td>7M</td>
    <td>512x512</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>Seg-T-Mask/16(With our attention loss)</td>
    <td>40.3</td>
    <td>7M</td>
    <td>512x512</td>
    <td><a href="https://drive.google.com/file/d/11RcrRDL0e_5i60kLPMiIcj2NLIkPJHvD/view?usp=drive_link/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_small_mask/variant.yml">config</a></td>
    <td><a href="https://drive.google.com/file/d/1ZI3IN85Uv67cW-6bvbcxiO13oY4-rot-/view?usp=drive_link">log</a></td>
  </tr>
</table>
We currently put the loss function code under the ‘Segmenter-master’ sub-folder. In the future, we will open a new repository to fully publish our work. ,

### Deployment friendly
Moreover, for better future practice, we further package the model into 'TDCN_deploy.pt'. 
Different from '.pth' file, this is a form 
that can save the model structure at the same time, 
which is more conducive to deployment.

Thanks to the torch.jit library integrated with our pytorch, 
you can directly download our compiled models. The link is https://drive.google.com/file/d/17fq2qsZmB2cL-QRIZH96x9PGunX_H8-W/view?usp=drive_link.

In addition, we have also added a new file named 'inference.py' for direct prediction.
The boundary can be predicted just by giving the path of the image, as shown below.
    
       python inference.py -filename demo.jpg
       python inference.py -filename demo.jpg  --cuda_available (if GPU available)

We believe that the newly added code may bring convenience to edge device deployment in the future.

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
Part of our code comes from [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf), [Pidinet Repository](https://github.com/zhuoinoulu/pidinet), [Segmenter Repository](https://github.com/rstrudel/segmenter). We are very grateful for these excellent works.
