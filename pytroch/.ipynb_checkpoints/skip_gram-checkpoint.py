import matplotlib.pyplot as plt

sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student",]
# 将所有句⼦连接在⼀起，然后⽤空格分隔成多个单词
words = '  '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建⼀个字典，将每个词映射到⼀个唯⼀的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建⼀个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list) # 计算词汇表的⼤⼩
print("词汇表：", word_list) # 输出词汇表
print("词汇到索引的字典：", word_to_idx) # 输出词汇到索引的字典
print("索引到词汇的字典：", idx_to_word) # 输出索引到词汇的字典
print("词汇表⼤⼩：", voc_size) # 输出词汇表⼤⼩

# ⽣成Skip-Gram训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = [] # 初始化数据
    for sentence in sentences: # 遍历句⼦
        sentence = sentence.split()  # 将句⼦分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将当前单词前后各N个单词作为相邻单词
            for neighbor in sentence[max(idx - window_size, 0): 
                        min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词与当前单词作为⼀组训练数据
                    data.append((word, neighbor))
    return data
# 使⽤函数创建Skip-Gram训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的Skip-Gram数据样例（前10个）
print("Skip-Gram数据样例（未编码）：", skipgram_data[:10])

# 定义One-Hot编码函数
import torch # 导⼊torch库
def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx)) # 创建⼀个⻓度与词汇表相同的全0张量  
    tensor[word_to_idx[word]] = 1  # 将对应词索引位置上的值设为1
    return tensor  # 返回⽣成的One-Hot编码后的向量
# 展示One-Hot编码前后的数据
word_example = "Teacher"
print("One-Hot编码前的单词：", word_example)
print("One-Hot编码后的向量：", one_hot_encoding(word_example, word_to_idx))

# 定义Skip-Gram类
import torch.nn as nn # 导⼊neural network
class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表⼤⼩到嵌⼊层⼤⼩（维度）的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)  
        # 从嵌⼊层⼤⼩（维度）到词汇表⼤⼩的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)  
    def forward(self, X): # 前向传播的⽅式，X形状为(batch_size, voc_size)      
            # 通过隐藏层，hidden形状为 (batch_size, embedding_size)
            hidden = self.input_to_hidden(X) 
            # 通过输出层，output_layer形状为 (batch_size, voc_size)
            output = self.hidden_to_output(hidden)  
            return output    
embedding_size = 2 # 设定嵌⼊层的⼤⼩，这⾥选择2是为了⽅便展示
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化Skip-Gram模型
print("Skip-Gram类：", skipgram_model)

# 训练Skip-Gram类
learning_rate = 0.001 # 设置学习速率
epochs = 1000 # 设置训练轮次
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim # 导⼊随机梯度下降优化器
optimizer = optim.SGD(skipgram_model.parameters(), lr=learning_rate)  
# 开始训练循环
loss_values = []  # ⽤于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0 # 初始化损失值
    for center_word, context in skipgram_data:        
        X = one_hot_encoding(center_word, word_to_idx).float().unsqueeze(0) # 将中⼼词转换为One-Hot向量   
        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long) # 将周围词转换为索引值 
        y_pred = skipgram_model(X)  # 计算预测值
        loss = criterion(y_pred, y_true)  # 计算损失
        loss_sum += loss.item() # 累积损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch+1) % 100 == 0: # 输出每100轮的损失，并记录损失
        print(f"Epoch: {epoch+1}, Loss: {loss_sum/len(skipgram_data)}")  
        loss_values.append(loss_sum / len(skipgram_data))
        # 绘制训练损失曲线
# 绘制⼆维词向量图
plt.plot(range(1, epochs//100 + 1), loss_values) # 绘图
plt.title('训练损失曲线') # 图题
plt.xlabel('轮次') # X轴Label
plt.ylabel('损失') # Y轴Label
plt.show() # 显示图

# 输出Skip-Gram习得的词嵌⼊
print("Skip-Gram词嵌⼊：")
for word, idx in word_to_idx.items(): # 输出每个词的嵌⼊向量
    print(f"{word}: {skipgram_model.input_to_hidden.weight[:,idx].detach().numpy()}")

fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    # 获取每个单词的嵌⼊向量
    vec = skipgram_model.input_to_hidden.weight[:,idx].detach().numpy() 
    ax.scatter(vec[0], vec[1]) # 在图中绘制嵌⼊向量的点
    ax.annotate(word, (vec[0], vec[1]), fontsize=12) # 点旁添加单词标签
plt.title('⼆维词嵌⼊') # 图题
plt.xlabel('向量维度1') # X轴Label
plt.ylabel('向量维度2') # Y轴Label
plt.show() # 显示图