import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = [] # 初始化一个空列表，用于存放每个簇的数据点
        # 循环self.cluster_num次，为每个簇初始化一个空列表
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray: # 遍历输入的数据集
            # 初始化最小距离和对应的索引
            distance_min = sys.maxsize
            index = -1
            # 遍历每个预定义的簇中心点。
            for i in range(len(self.points)):
                # 计算当前数据点与簇中心点的距离
                distance = self.__distance(item, self.points[i])
                # 如果找到一个更近的簇中心点
                if distance < distance_min:
                    distance_min = distance
                    index = i
            # 将当前数据点加入到对应的簇中
            result[index] = result[index] + [item.tolist()]
        # 初始化一个空列表，用于存放新的簇中心点。
        new_center = []
        # 遍历每个簇的数据点
        for item in result:
            # 计算并添加新的簇中心点。
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        # 如果新的簇中心点与旧的完全相同，说明达到稳态，结束递归
        if (self.points == new_center).all():
            # 计算所有数据点到其所属簇中心的距离之和
            sum = self.__sumdis(result)
            # 返回结果、簇中心点和距离之和。
            return result, self.points, sum
        # 更新簇中心点。
        self.points = np.array(new_center)
        # 递归调用cluster方法，继续迭代.
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    # 从给定的ndarray中随机选择指定数量的点作为起始点，并返回这些点的坐标值
    def __pick_start_point(self, ndarray, cluster_num):
        # 检查传入的cluster_num是否在有效范围内。如果不在，它会抛出一个异常，表示“簇数设置有误”。
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        # 这里使用random.sample函数从0到ndarray.shape[0] - 1的整数中随机选择cluster_num个不重复的整数。这些整数表示在ndarray中的下标。
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)

        points = []
        for index in indexes:
            # 从ndarray中提取每个点的坐标值。之后，它将每个坐标值转化为列表格式。
            points.append(ndarray[index].tolist())
        # 最后，将提取的点坐标值转换为NumPy数组格式并返回
        return np.array(points)

# 生成一个100x8的矩阵，其中的元素是从0到1的随机数。也就是说，我们有一个包含100个样本和每个样本有8个特征的数据集。
x = np.random.rand(100, 8)
# 创建一个KMeans聚类器对象，其中x是我们的输入数据，而10是我们要创建的簇的数量。
kmeans = KMeansClusterer(x, 10)
# 对数据进行聚类。并返回三个值
# result: 一个数组，表示每个数据点所属的簇的标签。
# centers: 一个数组，表示每个簇的中心点。
# distances: 一个数组，表示每个数据点到其所属簇中心的距离。
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)