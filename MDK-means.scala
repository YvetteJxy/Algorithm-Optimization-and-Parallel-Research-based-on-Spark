import CanopyKMeans.{closestPoint, generateInitialCenters, parseVector, scaleVector, squaredDistance, sumVectors}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector

import scala.util.control.Breaks._
import org.apache.spark.sql.SparkSession

object CanopyKMeans_test {
  // 将每一行字符串读取后，转换为向量
  def parseVector(line: String): Vector = {
    var values = line.split(",").map(_.toDouble)
    Vectors.dense(values) // 将values数组返回为密集向量的对象
  }

  // 计算两个向量之间的欧几里德距离的平方
  def squaredDistance(v1: Vector, v2: Vector): Double = {
    val diff = Vectors.dense(v1.toArray.zip(v2.toArray).map { case (a, b) => a - b }) // diff表示两个向量对应元素的差值
    Vectors.sqdist(diff, Vectors.zeros(diff.size)) //计算欧氏距离的平方
  }

  def closestPoint(p: Vector, centers: Array[Vector]): Int = {
    var bestIndex = 0
    var closest = Double.PositiveInfinity
    for (i <- 0 until centers.length) {
      val tempDist = squaredDistance(p, centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
    bestIndex
  }

  def sumVectors(v1: Vector, v2: Vector): Vector = {
    val sum = Vectors.dense(v1.toArray.zip(v2.toArray).map { case (a, b) => a + b })
    sum
  }

  def sumVectors1(v1: Vector, v2: Vector): Double = {
    val sum = Vectors.dense(v1.toArray.zip(v2.toArray).map { case (a, b) => a + b }).toArray.sum
    sum
  }

  def scaleVector(vector: Vector, scalar: Double): Vector = {
    val scaledValues = vector.toArray.map(_ * scalar)
    Vectors.dense(scaledValues)
  }

  // 使用最小最大原则优化Canopy算法的中心点选择
  def generateInitialCenters(data: Array[Vector], canopyT2: Double): Array[Vector] = {
    val rand = new scala.util.Random(20222132)

    // 计算数据集中所有样本点之间的距离和密度
    val distances = data.map(p => data.map(q => squaredDistance(p, q)))
    val densities = distances.map(_.count(_ <= canopyT2))

    // 找出密度最大的样本点作为第一个聚类中心，并从数据集中移除
    val maxDensityIndex = densities.zipWithIndex.maxBy(_._1)._2
    val initialCenter = data(maxDensityIndex)
    var data_array = data.filterNot(_ == initialCenter)
    var centers = Array(initialCenter)
    var j = 1
    for (point <- data_array if j < 3) {
      var included = false
      breakable {
        centers.foreach(c => {
          val distance = math.sqrt(squaredDistance(point, c))
          if (distance <= canopyT2) {
            data_array = data_array.filterNot(_ == point)
            included = true
            break
          }
        })
      }

      if (!included) {
        // 定义权重乘积为样本点密度、所在簇内样本平均距离倒数和簇间距离的乘积，选择权重乘积最大的样本点作为下一个聚类中心，并从数据集中移除
        val weights = data_array.map(p => {
          val clusterDistances = centers.map(c => math.sqrt(squaredDistance(p, c)))
          val minClusterDistanceIndex = clusterDistances.zipWithIndex.minBy(_._1)._2
          val minClusterDistance = clusterDistances(minClusterDistanceIndex)
          val clusterDensity = densities(minClusterDistanceIndex)
          val clusterAverageDistance = distances(minClusterDistanceIndex).sum / clusterDensity
          densities(maxDensityIndex) * (1 / clusterAverageDistance) * minClusterDistance
        })
        val maxWeightIndex = weights.zipWithIndex.maxBy(_._1)._2
        val nextCenter = data_array(maxWeightIndex)
        centers :+= nextCenter
        data_array = data_array.filterNot(_ == nextCenter)
        j = j + 1
      }
    }

    centers
  }

  def runCanopyKmeans(args: Array[String]): Unit = {
    val rand = new scala.util.Random(20222132)
    val conf = new SparkConf().setAppName("CanopyKmeans").setMaster("local[1]")
    val sc = new SparkContext(conf)

    // implicit val vectorEncoder: Encoder[Vector] = Encoders.kryo[Vector]

    val lines = sc.textFile(args(0), minPartitions = 5)
    val data = lines.map(parseVector).cache() // 数据预处理，向量化


    // 自适应canopy求t2
    // val data_arr = data.collect()
    var sum = 0.0
    for (pi <- data.collect()) {
      for (pj <- data.collect()) {
        sum += math.sqrt(squaredDistance(pi, pj))
      }
    }
    val t2 = math.sqrt(sum) * 0.011

    // 使用 Canopy 算法生成初始聚类中心
    val initialCenters = generateInitialCenters(data.takeSample(withReplacement = false, 70).toArray, t2)
    val K = initialCenters.length

    // 打印初始聚类中心，由canopy法求得
    println("Initial centers:")
    initialCenters.foreach(println)

    // 进行 KMeans 聚类

    // 从rdd中随机选择K个样本作为初始聚类中心，并将其转化为数组kPoints
    val kPoints = initialCenters // 初始聚类中心
    val convergeDist = args(2).toDouble // 收敛距离
    var tempDist = 1.0 // 初始化距离
    val MaxIteration = args(1).toInt // 初始化最大迭代次数
    var tempIteration = 0 // 初始化迭代次数
    while (tempDist > convergeDist && tempIteration < MaxIteration) {
      val closest = data.map(p => (closestPoint(p, kPoints), (p, 1))) // (中心点索引，（样本点，数量）)
      val pointStats = closest.reduceByKey {
        case ((v1, c1), (v2, c2)) => (sumVectors(v1, v2), c1 + c2)
      }
      val newPoints = pointStats.map { pair =>
        (pair._1, scaleVector(pair._2._1, 1.0 / pair._2._2))
      }.collectAsMap()


      tempDist = 0.0
      for (i <- 0 until K) {
        tempDist += squaredDistance(kPoints(i), newPoints(i))
      }
      for (newP <- newPoints) {
        kPoints(newP._1) = newP._2
      }
      println("Finished iteration(delta = " + tempDist + ")")
      println(tempIteration)
      tempIteration += 1
    }

    // 打印最终的聚类中心
    println("Final centers:")
    kPoints.foreach(println)

    sc.stop()
  }

  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    runCanopyKmeans(Array("E:\\校内课程\\分布式\\iris_noclass_blank.data", "10", "0.01"))
    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total runtime: " + totalTime + " ms")
  }
}