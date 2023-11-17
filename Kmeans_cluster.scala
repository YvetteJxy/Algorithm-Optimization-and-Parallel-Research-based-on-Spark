import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import breeze.linalg.DenseVector
import scala.util.Random
import breeze.linalg._
import org.apache.spark.mllib.linalg. Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
//val command = Array(
//  "--class",
//  "org.apache.spark.repl.Main",
//  "spark-shell")
//org.apache.spark.deploy.SparkSubmit.main(command)

// 此方法用Kmeans聚合
object Kmeans_cluster {
  // 随机种子
  val rand = new Random(20222132)

  // 将每一行字符串读取后，转换为向量
  def parseVector(line: String): Vector[Double] = {
    DenseVector(line.split(",").map(_.toDouble))
  }

  // 寻找给定向量p(某一个样本点)属于哪一个簇
  def closestPoint(p: Vector[Double], centers: Array[Vector[Double]]): Int = {
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

  def km_nodistribution(data: Array[Array[Double]], k: Int, maxIterations: Int, initialCenters: Array[Array[Double]]): Array[Array[Double]] = {
    val numFeatures = data.head.length

    // 迭代更新聚类中心
    var centers = initialCenters
    var iteration = 0

    while (iteration < maxIterations) {
      // 分配样本到最近的中心点所属的簇
      val assignments = data.map { point =>
        val vector = DenseVector(point: _*)
        val distances = centers.map(center => squaredDistance(DenseVector(center: _*), vector))
        val closestCenter = distances.zipWithIndex.minBy(_._1)._2
        closestCenter
      }

      // 更新每个簇的中心点
      val newCenters = Array.ofDim[Double](k, numFeatures)
      val clusterCounts = Array.fill(k)(0)

      for ((point, assignment) <- data.zip(assignments)) {
        for (i <- 0 until numFeatures) {
          newCenters(assignment)(i) += point(i)
        }
        clusterCounts(assignment) += 1
      }

      for (i <- 0 until k) {
        if (clusterCounts(i) != 0) {
          for (j <- 0 until numFeatures) {
            newCenters(i)(j) /= clusterCounts(i)
          }
        } else {
          // 处理空簇的情况，例如随机初始化中心点或选择其他策略
          // 这里给出一个示例：将空簇的中心点随机初始化为数据集中的一个样本点
          val randomIndex = rand.nextInt(data.length)
          newCenters(i) = data(randomIndex).toArray
        }
      }


      centers = newCenters
      iteration += 1
    }

    centers
  }


  def run(args: Array[String]): Unit = {
    // 单机模拟集群，此处使用3个线程模拟集群
    val conf = new SparkConf().setAppName("Kmeans").setMaster("local[1]")  // 线程
    val sc = new SparkContext(conf)
    val lines = sc.textFile(args(0),5)  // 读取数据，并进行分区
    val K = args(1).toInt   // K，聚类的簇数
    val convergeDist = args(2).toDouble   // 收敛距离
    val data = lines.map(parseVector).cache()    // 数据预处理，向量化

    // 从rdd中随机选择K个样本作为初始聚类中心，并将其转化为数组kPoints
    var kPoints = data.takeSample(withReplacement = false, K, 20222132)  // 随机选取初始聚类点，此处在改良方法中用来传参

    for (point <- kPoints) {
      println("坐标: " + point.toArray.mkString(", "))
    }

    var tempDist = 1.0
    val MaxIteration = args(3).toInt
    var tempIteration = 0

    // 迭代求解新的中心点
    while (tempDist > convergeDist && tempIteration < MaxIteration) {
      val closest = data.map(p => (closestPoint(p, kPoints), (p, 1)))  // 局部 （中心点索引，（样本点加和，数量））
      val pointStats = closest.reduceByKey {
        case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)
      }
      val newPoints = pointStats.map { pair =>
        (pair._1, pair._2._1 * (1.0 / pair._2._2))   // 局部新的中心点  （中心点索引，新的中心点坐标）
      }
      // 将newPoints在各个分区中的结果聚合到总机,data_temp 还是一个rdd
      val data_temp = newPoints.map { case (_, coordinates) =>   // data_temp就是各个中心点的坐标汇总起来
        coordinates
      }
//      val data_final = data_temp.coo
      // 计算所有样本点两两距离最短的 K 对样本点的中点
      val pairwiseDistances = data_temp.cartesian(data_temp).filter { case (point1, point2) =>
        point1 != point2
      }.map { case (point1, point2) =>
        val distance = Vectors.sqdist(Vectors.dense(point1.toArray), Vectors.dense(point2.toArray))
        (point1, point2, distance)
      }.sortBy(_._3).take(K)   // 找到K对距离最近的样本点，数据结构为 (point1, point2, distance)

      val center_set = data_temp.collect()

      // 提取中点作为初始中心点
      val initialCenters = pairwiseDistances.map { case (point1, point2, _) =>
        Vectors.dense(Array((point1.toArray, point2.toArray).zipped.map(_ + _).map(_ / 2): _*))
      }

      // 运行 K-means 算法进行聚类,调用函数

      var new_Centers_temp: Array[Array[Double]]= km_nodistribution(center_set.map(x => x.toArray),K,10,initialCenters.map(x => x.toArray))

      val new_Centers: Array[Vector[Double]] = new Array[Vector[Double]](new_Centers_temp.length)
      for (i <- new_Centers_temp.indices) {
        new_Centers(i) = DenseVector(new_Centers_temp(i): _*)
      }


      kPoints = new_Centers.clone()

      tempDist = 0.0
      for (i <- 0 until K) {
        tempDist += squaredDistance(kPoints(i), new_Centers(i))
      }
      println("Finished iteration(delta = " + tempDist + ")")
      tempIteration += 1
    }
    println("Final centers:")
    kPoints.foreach(println)
    sc.stop()
  }
}

object Main1 {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    Kmeans_cluster.run(Array("E:\\研一下\\分布式计算\\final\\datasets\\wine_noclass_blank.data", "3", "0.01", "10"))
    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total runtime: " + totalTime + " ms")
  }
}
