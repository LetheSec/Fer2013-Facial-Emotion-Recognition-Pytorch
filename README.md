## Fer2013 - Facial Emotion Recognition

This work is the final project of the Computer Vision Course of USTC. However, I achieve the highest single-network
classification accuracy on FER2013 based on ResNet18. To my best knowledge, this work **achieves state-of-the-art
single-network accuracy of 73.70 % on FER2013 without using extra training data**, which exceeds the previous work [1]
of 73.28%.

|  Method   | Private Test Data |
| :-------: | :---------------: |
|    [1]    |      73.28%       |
| This work |    **73.70%**     |


Official model checkpoint and training log can be found following:

- [Google Drive](https://drive.google.com/file/d/1H4nW58EmGnDr3R7mQ6WM820bYLBpFagG/view?usp=sharing)
- [USTC Cloud Disk](https://rec.ustc.edu.cn/share/3b429380-687a-11ec-a232-e55771970099)
- [百度网盘 (提取码: h927)](https://pan.baidu.com/s/1yh-mcD4MAmDj-yltCMqbFw)


Fer2013 Leaderboard: [Here](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013)

<img src="https://raw.githubusercontent.com/LetheSec/oss/master/uPic/MhTbhC.png"  width="700" />


### Usage

First, you should download the
official [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz)
dataset, and place it in the outmost folder with the following folder structure `datasets/fer2013/fer2013.csv`

To train your own model, run the following:

```
python train.py --name='your_version'
```

To evaluate the model, run the follwing

```
python evaluate.py --checkpoint='xxx/best_checkpoint.tar'
```

### Result

<img src="https://raw.githubusercontent.com/LetheSec/oss/master/uPic/XpQTQ8.png"  width="500" />

### Reference

[1] Khaireddin, Yousif, and Zhuofa Chen. "Facial Emotion Recognition: State of the Art Performance on FER2013." arXiv
preprint arXiv:2105.03588 (2021).

