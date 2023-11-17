import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import scala.util.Random
import java.util.concurrent.ThreadLocalRandom
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.storage.StorageLevel

// 此法用于collectAsMap方法聚合
object Kmeans {
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

  def run(args: Array[String]): Unit = {
    // 单机模拟集群，此处使用3个线程模拟集群
    val conf = new SparkConf().setAppName("Kmeans").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val lines = sc.textFile(args(0),5)  // 读取
    val K = args(1).toInt   // K，聚类的簇数
    val convergeDist = args(2).toDouble   // 收敛距离
    val data = lines.map(parseVector).cache()    // 数据预处理，向量化

    // 从rdd中随机选择K个样本作为初始聚类中心，并将其转化为数组kPoints
    var kPoints = data.takeSample(withReplacement = false, K, 20222132).toArray   // 随机选取初始聚类点，此处在改良方法中用来传参
    // println(kPoints)
    var tempDist = 1.0
    val MaxIteration = args(3).toInt
    var tempIteration = 0
    // 迭代求解新的中心点
    while (tempDist > convergeDist && tempIteration < MaxIteration) {
      val closest = data.map(p => (closestPoint(p, kPoints), (p, 1)))   // (中心点索引，（属于该中心的样本，个数）)
      val pointStats = closest.reduceByKey {
        case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)   // (中心点索引，（属于该中心的样本加和，个数）)
      }
      val newPoints = pointStats.map { pair =>
        (pair._1, pair._2._1 * (1.0 / pair._2._2))     //  （中心点索引，更新后的中心点）
      }.collectAsMap()

      tempDist = 0.0
      for (i <- 0 until K) {
        tempDist += squaredDistance(kPoints(i), newPoints(i))
      }
      for (newP <- newPoints) {
        kPoints(newP._1) = newP._2
      }
      println("Finished iteration(delta = " + tempDist + ")")
      tempIteration += 1
    }
    println("Final centers:")
    kPoints.foreach(println)
    sc.stop()
  }
}

object Main {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    Kmeans.run(Array("E:\\研一下\\分布式计算\\final\\datasets\\wine_noclass_blank.data", "3", "0.01", "100"))
    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total runtime: " + totalTime + " ms")
  }
}
