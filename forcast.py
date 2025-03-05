import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import matplotlib.pyplot as plt#导入强大的绘图库
import torch#一个深度学习的库Pytorch
import torch.nn as nn#neural network,神经网络
import torch.optim as optim#一个实现了各种优化算法的库
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别.
#设置随机种子
import random
torch.backends.cudnn.deterministic = True#将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False#关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
train_df=pd.read_csv("ocean_observation_data.csv")
print(f"len(train_df):{len(train_df)}")
train_df.head()

tp=train_df['Temperature'].values
print(f"len(tp):{len(tp)}")
from sklearn.preprocessing import MinMaxScaler
# 创建MinMaxScaler对象
scaler = MinMaxScaler()
# 将数据进行归一化
tp = scaler.fit_transform(tp.reshape(-1,1))
def split_data(data,time_step=12):
    dataX=[]
    datay=[]
    for i in range(len(data)-time_step):
        dataX.append(data[i:i+time_step])
        datay.append(data[i+time_step])
    dataX=np.array(dataX).reshape(len(dataX),time_step,-1)
    datay=np.array(datay)
    return dataX,datay
dataX,datay=split_data(tp,time_step=12)
print(f"dataX.shape:{dataX.shape},datay.shape:{datay.shape}")
#划分训练集和测试集的函数
def train_test_split(dataX,datay,shuffle=True,percentage=0.8):
    """
    将训练数据X和标签y以numpy.array数组的形式传入
    划分的比例定为训练集:测试集=8:2
    """
    if shuffle:
        random_num=[index for index in range(len(dataX))]
        np.random.shuffle(random_num)
        dataX=dataX[random_num]
        datay=datay[random_num]
    split_num=int(len(dataX)*percentage)
    train_X=dataX[:split_num]
    train_y=datay[:split_num]
    test_X=dataX[split_num:]
    test_y=datay[split_num:]
    return train_X,train_y,test_X,test_y
train_X,train_y,test_X,test_y=train_test_split(dataX,datay,shuffle=False,percentage=0.8)
print(f"train_X.shape:{train_X.shape},test_X.shape:{test_X.shape}")
X_train,y_train=train_X,train_y


# 定义CNN+LSTM模型类
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Conv1d(conv_input, conv_input, 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化记忆状态c0
        # print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out


test_X1 = torch.Tensor(test_X)
test_y1 = torch.Tensor(test_y)

# 定义输入、隐藏状态和输出维度
input_size = 1  # 输入特征维度
conv_input = 12
hidden_size = 64  # LSTM隐藏状态维度
num_layers = 5  # LSTM层数
output_size = 1  # 输出维度（预测目标维度）

# 创建CNN_LSTM模型实例
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size)

# 训练周期为500次
num_epochs = 500
batch_size = 64  # 一次训练的数量
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
# 损失函数
criterion = nn.MSELoss()

train_losses = []
test_losses = []

print(f"start")

for epoch in range(num_epochs):

    random_num = [i for i in range(len(train_X))]
    np.random.shuffle(random_num)

    train_X = train_X[random_num]
    train_y = train_y[random_num]

    train_X1 = torch.Tensor(train_X[:batch_size])
    train_y1 = torch.Tensor(train_y[:batch_size])

    # 训练
    model.train()
    # 将梯度清空
    optimizer.zero_grad()
    # 将数据放进去训练
    output = model(train_X1)
    # 计算每次的损失函数
    train_loss = criterion(output, train_y1)
    # 反向传播
    train_loss.backward()
    # 优化器进行优化(梯度下降,降低误差)
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            output = model(test_X1)
            test_loss = criterion(output, test_y1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch:{epoch},train_loss:{train_loss},test_loss:{test_loss}")
def mse(pred_y,true_y):
    return np.mean((pred_y-true_y) ** 2)
train_X1=torch.Tensor(X_train)
train_pred=model(train_X1).detach().numpy()
test_pred=model(test_X1).detach().numpy()
pred_y=np.concatenate((train_pred,test_pred))
pred_y=scaler.inverse_transform(pred_y).T[0]
true_y=np.concatenate((y_train,test_y))
true_y=scaler.inverse_transform(true_y).T[0]
print(f"mse(pred_y,true_y):{mse(pred_y,true_y)}")

train_losses = [loss.detach().numpy() for loss in train_losses]

# 生成横坐标数据，假设训练轮次数量就是train_losses列表的长度
step_numbers = np.arange(0, len(train_losses) * 50, 50)

# 绘制训练损失曲线
plt.plot(step_numbers, train_losses, label='Train Loss', marker='o')
# 设置标题、坐标轴标签以及图例等
plt.title('Train Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.title("CNN_LSTM")
x=[i for i in range(len(true_y))]
plt.plot(x,pred_y,marker="o",markersize=1,label="pred_y")
plt.plot(x,true_y,marker="x",markersize=1,label="true_y")
plt.legend()
plt.show()
new_data=np.arange(800,812).reshape(-1,1)
new_data_tensor = torch.Tensor(new_data)

# 创建模型实例（这里假设你已经有了一个自定义的包含CNN和LSTM结构的模型类，比如名为CNN_LSTM，下面是简单示意初始化）
model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size)


# 进行预测，依次经过卷积层（假设的CNN部分）和LSTM层以及后续可能的处理来获取最终预测结果
with torch.no_grad():
    x = new_data_tensor
    # 经过卷积层（假设模型中有卷积层处理在前，这里简单示意卷积操作，根据实际模型结构调整）
    if hasattr(model, 'conv'):
        x = model.conv(x)
        # 如果卷积层输出维度等不符合后续LSTM输入要求，可能需要调整维度，比如下面这样（示例，根据实际调整）
        x = x.permute(2, 0, 1)  # 调整维度顺序以匹配LSTM输入要求（假设需要这样调整，根据实际模型定）
    # 确保经过卷积层后的数据形状符合LSTM输入期望，这里简单打印查看一下形状（调试用，可去掉）
    print("Shape of x before LSTM:", x.shape)
    # LSTM前向传播
    out, _ = model.lstm(x, (h0, c0))
    # 如果模型后续还有其他层进行处理，比如全连接层来得到最终输出维度，这里简单示意（根据实际调整）
    if hasattr(model, 'fc'):  # 假设模型有全连接层fc用于输出维度转换
        out = model.fc(out)
    predictions = out
    predictions = predictions.detach().numpy()

print(predictions)