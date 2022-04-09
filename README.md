# crack-classify
### 数据集划分
data目录演示
```
tree -d
.
├── raw
│    ├── Negative
│    └── Positive
├── small
│    ├── Negative
│    └── Positive
├── test
│    ├── Negative
│    └── Positive
├── train
│    ├── Negative
|    └── Positive
└── valid
     ├── Negative
     └── Positive
```
数据集地址[https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)  
将数据集拷贝到data目录下修改文件夹名为raw，运行split_data.py，即可按6:2:2随机生成train、valid、test数据集
```shell
python3 split_data.py
```

### 训练与测试模型
训练命令
```shell
python3 train.py --dataset ./data/train/ --img_size 256 --batch_size 8 --epochs 100 --model_path ./model/model.pt  --label_path ./data/train/label.txt
```
测试命令
```shell
python3 test.py --dataset ./data/test/ --img_size 256 --batch_size 8 --model_path ./model/model.pt
```
