# CCPM-baseline

本项目为北京大学2021年秋季学期计算语言学课程项目，古诗文识记数据集的基线模型代码

本方案将该任务建模成 Multi-choice QA 任务, 利用 RoBERTa 作为 backbone 进行 fine-tuning, 供同学们参考。

训练集：21,778句；验证集：2,720 句；测试集：2,720 句。

数据集来自 BAAI CUGE benchmark:

> @article{li2021CCPM,
> 
> title = {CCPM: A Chinese Classical Poetry Matching Dataset},
> 
> author = {Li, Wenhao and Qi, Fanchao and Sun, Maosong and Yi, Xiaoyuan and Zhang, Jiarui},
> 
> journal={arXiv preprint arXiv:2106.01979},
> 
> year = {2021}
> }


## Environment Setup

推荐使用 Anaconda 进行虚拟环境的配置，命令如下：
```bash
conda create -n ccpm python=3.7
conda activate ccpm 
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install transformers datasets 
```

## Training
环境设置完毕后,使用提供的 `run.sh` 脚本即可启动训练和在验证集上进行验证, 结果如下:
> ***** eval metrics *****
> 
>  epoch                   =        3.0
>  
>  eval_accuracy           =     0.8676
>  
>  eval_loss               =     0.4276
>  
>  eval_runtime            = 0:00:05.38
