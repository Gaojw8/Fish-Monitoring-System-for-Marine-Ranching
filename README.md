# Fish-Monitoring-System-for-Marine-Ranching
The `Fish Monitoring System for Marine Ranching` uses computer vision to detect and count fish, measuring activity levels based on fish density over time. This helps monitor fish behavior, optimize feeding, and assess ecosystem health, improving aquaculture efficiency and sustainability.
面向海洋牧场的鱼类监测系统利用计算机视觉进行鱼类目标检测和计数，通过单位时间内的鱼类数量反映其活动水平。该系统有助于监测鱼类行为、优化投喂策略，并评估生态健康，提高水产养殖效率和可持续性。

1.平台与数据集准备
平台：Google colab
数据集：采用Kaggle上的鱼类开源数据集[https://www.kaggle.com/datasets/markdaniellampa/fish-dataset/cod](https://www.kaggle.com/code/zehraatlgan/fish-detection-with-yolo11)e
代码框架：tensorflow

2.colab平台下载数据集
首先登陆Kaggle的Api：
`import json
token = {"username":"gaojun123","key":"bef9281dc9d8baf163062f4c7e7b0e5c"}
with open('/content/kaggle.json', 'w') as file:
  json.dump(token, file)
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle config set -n path -v /content`
进行数据集下载：
`import kagglehub
path = kagglehub.dataset_download("zehraatlgan/fish-detection")
print("Path to dataset files:", path)`
数据集分布如下：
![image](https://github.com/user-attachments/assets/1b3e5892-93a3-4c85-8f2f-b4f3b48973b6)

3.训练模型
`%pip install ultralytics supervision roboflow`
!yolo task=detect mode=train model=yolo11s.pt data='/content/1/data.yaml' epochs=30 imgsz=640 plots=True

4.在监测系统中运行模型
将训练好的`best.pt`文件放在`/models/detect`文件夹下，运行`main.py`文件，打开监测界面，
![image](https://github.com/user-attachments/assets/2f3cb974-a2fd-4e25-b7b7-190d21c4291e)
选择监测模式，即可使用训练好的模型进行鱼类监测。
