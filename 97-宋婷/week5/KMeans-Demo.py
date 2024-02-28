import numpy as np
import random
import sys

'''
Kmeans 算法实现
'''

# k 均值聚类
class KMeansDemo:
    def __init__(self,ndarray,cluster_num):
        self.ndarray = ndarray
        self.cluseter_num = cluster_num
        self.points = self.__pick_start_point(ndarray,cluster_num)


    def cluster(self):
        # 初始化一个空列表 用于存放每个簇的数据点
        result = []
        # 循环self.cluster_num 为每个簇初始化一个空列表
        for i in range(self.cluseter_num):
            result.append([])
        # 遍历输入的数据集
        for item in self.ndarray:
            # 初始化最小距离和对应索引
            distance_min = sys.maxsize
            index = -1
            # 遍历每个预定义的簇中心点
            for i in range(len(self.points)):
                # 计算当前数据点与簇中心点的距离
                distance = self.__distance(item, self.points[i])
                # 如果找到一个更近的簇中心点
                if distance < distance_min:
                    distance_min = distance
                    index = i

            # 将当前数据点加入对应的簇中
            result[index] = result[index] + [item.tolist()]
        # 初始化一个空列表 用于存放新的簇中心点
        new_center = []
         # 遍历每个簇的数据点
        for item in result:
            # 计算并添加新的簇中心点
            new_center.append(self.__center(item).tolist())
        # 如果新的中心点与旧的完全相同 说明达到稳态 结束递归
        if(self.points == new_center).all():
            # 计算所有数据点到其所属簇中心点的距离之和
            sum = self.__sumdis(result)
            # 返回结果、簇中心点和距离之和
            return result,self.points,sum

        # 更新簇中心点
        self.points = np.array(new_center)
        # 递归调用cluster方法，继续迭代
        return self.cluster()

    def __sumdis(self,result):
        # 计算总距离之和
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+= self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return  np.array(list).mean(axis=0)

    def __distance(self,p1,p2):
        # 计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp,0.5)

    # 从给定的ndarray中随机选择指定数量的点作为起始点，并返回这些点的坐标值
    def __pick_start_point(self,ndarray,cluster_num):
        # 检查传入的cluser_num是否在有效范围内
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0,ndarray.shape[0], step=1).tolist(), cluster_num)

        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

x = np.random.rand(100,8)
kmeans = KMeansDemo(x,10)
resule, centers, distances = kmeans.cluster()
print(resule)
print(centers)
print(distances)