# DAN-2018: Transferable Representation Learning with Deep Adaptation Networks

## 1. Introduction

This is a PyTorch re-implementation of paper [DAN-2018](https://ieeexplore.ieee.org/abstract/document/8454781/authors#authors). This project was built on [TLlib](https://github.com/thuml/Transfer-Learning-Library) with few original scripts modified to support the DAN-2018 scenario.

## 2. Contents

+ [Installation](#3. Installation)
+ [How to Use](#4. How to Use)
+ [Results and Discussions](#5. Results and Discussions)
+ [Insights](#6. Insights)

## 3. Installation 

1. Windows + Python 3.10 (Linus should also be fine as long as it's Python 3.10)
2. Clone or download the repository
3. cd to the root directory of the cloned/downloaded repository
4. `pip install -r requirements.txt`

## 4. How to Use 

+ Go to `DAN_{model_name}.py`
+ Scroll down to find the `class Args` definition under `if __name__ == '__main__':`
+ Modify the parameters in `class Args` according to your situation
+ Note: If you are unsure what are the parameters about, simply check out the comments for the parameters

## 5. Results and Discussions

### 5.1 Term Explanation:

| Shortened Term | Explanation            |
| -------------- | ---------------------- |
| E-mi           | Entropy Minimization   |
| Mk-mi          | Mk-MMD Minimization    |
| Mk-ma          | Mk-MMD Maximization    |
| ML             | Multi-Layer Adaptation |

### 5.2 AlexNet

+ Amazon (2817 images) → Webcam (795 images)

|           Methods | E-mi? | Mk-mi? | Mk-ma? | ML?  | Best Acc |
| ----------------: | :---: | :----: | :----: | :--: | -------- |
|           AlexNet |   ×   |   ×    |   ×    |  ×   | 53       |
|              E-mi |   √   |   ×    |   ×    |  ×   | 52.7     |
| E-mi & Mk-mi & ML |   √   |   √    |   ×    |  √   | 62       |
|        Mk-mi & ML |   ×   |   √    |   ×    |  √   | 58.7     |
|               DAN |   √   |   √    |   √    |  √   | 64.4     |
|           DAN-fc7 |   √   |   √    |   √    |  ×   | 61.3     |

#### Discussions

1. **(AlexNet)**: The original paper reported that the accuracy of AlexNet from A to W is 60.6±0.5. I cannot obtain a similar result, and the best result I could ever get by manual parameter tunning is around 55.
2. **(E-mi)**: When we add only Entropy Minimization to the final loss, we could see that the result is no good than the original AlexNet.
3. **(E-mi & Mk-mi & ML)**: But when we add Entropy Minimization and Mk-MMD Minimization to the final loss, there could be a significant performance boost. The final accuracy is 62.
4. **(Mk-mi & ML)**: When the Entropy Minimization term is removed from the final loss, leaving Mk-MMD minimization alone, the accuracy drops from 62 to 58.7. Point 3 and 4 show the importance of applying Entropy Minimization and Mk-MMD Minimization simultaneously.
5. **(DAN)**: DAN achieves the best result of course (accuracy: 64.4), with an accuracy boost about 11 compared to the original AlexNet.
6. **(DAN-fc7)**: DAN-fc7 (only the fc7 layer is Mk-MMD-adapted) is slightly degraded from DAN, showing the importance of Multi-Layer Adaptation. 
7. **(Why Entropy Minimization Alone Doesn't Work):** When Mk-MMD Minimization is not include in the final loss function, the source and target domain will not be aligned. Thus, feeding the network with target samples and backpropagation based on Entropy Minimization loss will only affect the classifier on source domain, and how a classifier purely trained on the source domain will perform on a new domain that is much different is unpredictable (just like row 2 AlexNet). In all, without Mk-MMD minimization, the source domain and target domain is "far away" and all operations will only affect model's performance on source domain. (above is only my humble understanding)

### 5.3 ResNet-18

+ Amazon (2817 images) → Webcam (795 images)

|   Methods | E-mi? | Mk-mi? | Mk-ma? | ML?  | Best Acc |
| --------: | :---: | :----: | :----: | :--: | :------: |
| ResNet-18 |   ×   |   ×    |   ×    |  ×   |  72.579  |
|       DAN |   √   |   √    |   √    |  √   |  85.786  |

#### Discussions

1. **(ResNet-18):** Compared to the original paper, ResNet-50 achieved an accuracy of 68.4. Yet in my re-implementation, I use ResNet-18 to achieve a better accuracy of 72.6. This implies that when we already have a model powerful enough on the source domain, adding more blocks to the model to make it deeper doesn't guarantee better transferability. As the features extracted by the network appears to transition from general to specific as the data flows from input to the output of the network. Deeper doesn't always mean better.
2. **(DAN):** The result obtained by me (85.8) is quite close to the original paper (86.3) but it doesn't exceed 86.3 as inferred (Why I think it would exceed 86.3? Because ResNet-18 outperforms ResNet-50, and it would be a linear thinking that a DAN built on ResNet-18 would also outperform a DAN built on ResNet-50). 

## 6. Insights

1. **Q:** When calculating β, which constraint shall we use? 
   $$
   \sum_{u=0}^m\beta_u=1, \beta_u\geq0 \tag{1}
   $$
   or:
   $$
   M^T\beta_l=1, \beta_l\geq0\tag{2}
   $$
   **A:** When I train the network, I tried both constraints. The overall performance under constraint $(2)$ is always better than that under constraint $(1)$. 

   But, I have to set `transfer_loss = torch.abs(transfer_loss)` after the calculation of Mk-MMD, otherwise a large batches' Mk-MMD loss values will be negative and the overall performance will be influenced.

   Which leads to the following questions:

2. **Q:** Can the value of Mk-MMD be negative? If yes, what does it mean when Mk-MMD is negative?

   **A:** According to the empirical estimate of Mk-MMD:
   $$
   M_k(\mathcal{D}_s, \mathcal{D}_t) \triangleq \frac{1}{n_s^2} \sum_{i=1}^{n_s} \sum_{j=1}^{n_s} k(x_i^s, x_j^s) + \frac{1}{n_t^2} \sum_{i=1}^{n_t} \sum_{j=1}^{n_t} k(x_t^s, x_t^s) - \frac{2}{n_s n_t} \sum_{i=1}^{n_s} \sum_{j=1}^{n_t} k(x_i^s, x_j^t)
   $$
   For sure it can be negative, but what does it mean when it becomes negative? For the common case, it is negative when the first two terms is smaller then the third term.

   The first term measures the embedding distance between layer outputs of the source samples.

   The second term measures the embedding distance between layer outputs of the target samples.

   The third term measures the embedding distance between layer outputs between the source and the target samples.

   Being negative is actually a good result we seek, meaning that the embedding distance between samples from the same distribution is small enough and at the mean time, the embedding distance between samples from different distributions is big enough. Thus, the source and target domains are totally confused.

   Yet I think it is best to set the Mk-MMD to a small value (or 0) when its value is negative.

   But ↓

3. **Q:** Why `transfer_loss = torch.abs(transfer_loss)` will make the overall result better?

   **A:** I don't quite know the why, maybe it's only me being lucky, but I've got some guesses:

    + When the Mk-MMD loss is negative, it will affect the overall loss value, causing the network parameters to go to a different direction.
    + When the Mk-MMD loss is negative, the only way for it to not influence the overall loss value is to set it to zero or to a small value (Alert: Untested Theory!). And the Mk-MMD values are often small, thus the effect of taking absolute value is somehow equal to setting it small.

## Appendix

#### Which Library Scripts Did I Modified and Where

1. In `tlib/modules/classifier.py`, at line 78, from original like this:

   ```python
     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       """"""
       f = self.pool_layer(self.backbone(x))
       f = self.bottleneck(f)
       predictions = self.head(f)
       if self.training:
         return predictions, f
       else:
         return predictions
   ```

   to this, with the purpose of exposing the output of pooling layer (which will be adapted according to the original paper).

   ```python
     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """"""
       pool_layer_opt = self.pool_layer(self.backbone(x))
       bottleneck_opt = self.bottleneck(pool_layer_opt)
       head_opt = self.head(bottleneck_opt)
       if self.training:
         return pool_layer_opt, bottleneck_opt, head_opt
       else:
         return head_opt
   ```

2. In `tlib/alignment/dan.py`, at line 78, from original like:

   ```python
       def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
           features = torch.cat([z_s, z_t], dim=0)
           batch_size = int(z_s.size(0))
           self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


           kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
           # Add 2 / (n-1) to make up for the value on the diagonal
           # to ensure loss is positive in the non-linear version
           loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

           return loss
   ```

   to this, with the weights β incorporated into the final computation.

   ```python
       def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, beta: list) -> torch.Tensor:
           features = torch.cat([z_s, z_t], dim=0)
           batch_size = int(z_s.size(0))
           self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


           # kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
           kernel_matrix = sum([kernel(features) * Beta_u for kernel, Beta_u in zip(self.kernels, beta)])
           # Add 2 / (n-1) to make up for the value on the diagonal
           # to ensure loss is positive in the non-linear version
           loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

           return loss
   ```

   ​