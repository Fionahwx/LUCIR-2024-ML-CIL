
# Learning a Unified Classifier Incrementally via Rebalancing


大三下机器学习课程大作业，基于持续学习的图像分类，复现 _Learning a Unified Classifier Incrementally via Rebalancing_ 。论文地址：[LUCIR-paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)
### Requirements

`pip install requirements.txt`

### Running
`python main.py --args`

### Hyper-parameters

`--dataset` Dataset [CIFAR100] 【数据集】

`--start` Number of classes of first task 【初始任务的类别数】

`--increment` Number of classes at each next task 【每轮任务增加的类别数】

`--rehearsal` Number of example stored per each class 【每轮保存的旧样本数量】

`--selection` Selection of exemplar [Herding, Random, Closest to Mean] 【如何选择旧样本】

`--exR` if True, Exemplar are stored  【是否使用经验重放】

`--class_balance_finetuning` if True, a class balance fine-tuning is performed at the end of each task  【是否使用余弦归一化】

`--less_forg`  if True,  _**less-forget**_ constraint is used  【是否使用较小遗忘损失】

`--lambda_base` weight factor of less-forget loss  【较小遗忘损失权重】

`--ranking` if True,  _**margin ranking loss**_ constraint is used  【是否使用边缘排序损失】
### Comparison with original results

#### CIFAR-100

<table>
  <tr>
    <td>Starting Classes</td>
    <td>Increment</td>
    <td colspan="2">Average Incremental Accuracy</td>
  </tr>
  <tr>
    <td>50</td>
    <td>50</td>
    <td>68.12</td>
    
  </tr>

  <tr>
    <td>50</td>
    <td>10</td>
    <td>57.08</td>
    
  </tr>

  <tr>
    <td>50</td>
    <td>5</td>
    <td>52.33</td>
    
  </tr>
</table>


#### 消融实验

<table>
  <tr>
    <td colspan="2">Condition</td>
    <td>Starting Classes</td>
    <td>Increment</td>
    <td colspan="2">Average Incremental Accuracy</td>
  </tr>
  <tr>
    <td>CN</td>
    <td> </td>
    <td>50</td>
    <td>10</td>
    <td>55.91</td>
    
  </tr>

  <tr>
    <td>CN+LS</td>
    <td> </td>
    <td>50</td>
    <td>10</td>
    <td>61.38</td>
    
  </tr>

  <tr>
    <td>CN+LS+IS</td>
    <td> </td>
    <td>50</td>
    <td>10</td>
    <td>63.59</td>
    
  </tr>
</table>
