import breeze.linalg.{DenseVector, _}
import org.apache.spark.SparkContext.getOrCreate
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{HashSet, Map}
//import org.apache.spark.sql.SparkSession
//import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
//import org.sparkproject.dmg.pmml.True
//import org.apache.spark.storage.StorageLevel
//import scala.collection.mutable.ListBuffer
//import scala.collection.mutable.ArrayBuffer
//
//val command = Array(
//  "--class",
//  "org.apache.spark.repl.Main",
//  "spark-shell")
//org.apache.spark.deploy.SparkSubmit.main(command)


object Canopy_brief {


  //TODO:这里是函数定义;最好将全局变量定义也写在这里

  //这里定义的canopy_1返回的是此点是否能成为局部中心点，若可以则返回该点坐标，否则返回null
  var list = new HashSet[DenseVector[Double]]()
  // 第一个函数canopy_1
  def canopy_1(x: DenseVector[Double], T1: Double): Any = {
    if (list.isEmpty) { //若此时为初始化HashSet，即局部Canopy中心点列表为空，将该点加入列表作为中心点
      list=list + x
      return x
    } else {
     val Dist=list.map(center=>norm(center-x))
      //欧几里得距离就是L2范数，用map计算x与所有中心点的L2范数，并collect返回
      if (Dist.min > T1){
        list=list + x
        return x
      }
      return null   //说明该点x不能成为中心点
      }//if
    }//def canopy_1

  //第二个函数Canopy_2 : 对第一个函数的略微修改，为获得全局Canopy中心点列表

  //FiXME: 后续优化可以考虑，随机选择第一个中心点，或者用密度法选择第一个中心点；FIXME标注了大致要修改的地方
  def canopy_2(Mapcenters:List[DenseVector[Double]],T1:Double): List[(Int,DenseVector[Double])]= {
    var Fi_centers = new HashSet[(Int, DenseVector[Double])]()
    //这里需要使Fi_centers 有对应的序号index及其canopy center 向量
    var count:Int= 0 //为便于标记
    for (i <- 0 until Mapcenters.size) {
      if (Fi_centers.isEmpty) { //若此时为初始化HashSet，即全局Canopy中心点列表为空，将该点加入列表作为中心点
        count=count+1
        Fi_centers = Fi_centers + ((count, Mapcenters(i) ) )//FIXME
      } else {
        val Dist = Fi_centers.map(center => norm(center._2 - Mapcenters(i)))
        if (Dist.min > T1) {
          count=count+1
          Fi_centers = Fi_centers + ((count, Mapcenters(i)))
        }
      } //if
    } //for
    return Fi_centers.toList
  }

  //定义的第3个函数，用于返回数据点对应划分的（canopy index，point） 列表
  // （canopy index，point） 处理成了一个键制对形式，方便后续与cluster 中心作canopy index连接
def canopy_3( point:DenseVector[Double], Canopylist:List[(Int,DenseVector[Double])],T2:Double  ):List[(Int,DenseVector[Double])]={
  //计算各数据对象和Canopy全局中心点的距离，距离小于T2则将其加入该Canopy子集
  //val point=DenseVector(1.0,2.0,3.0,4.0)
  //val Canopylist= CanopyCenterList
  //这里由于传入的Canopy是一个List，不是RDD，即便用map也不会作并行计算。但是由于Canopy的中心点数量本身就不多，不用并行计算应该也不太影响速度
  val dist = Canopylist.map(center => (center._1,norm(center._2 - point)  ) )
  val index=dist.filter(x=>x._2<T2).map(y=>(y._1,point))  //注意filter也是转化操作，还是返回RDD,但collect之后就不是了
  index // （index，point）List ,index为point对应的Canopy键
}

  //定义的第4个函数，用于返回聚类中心点对应划分的（canopy index，(center,index)） 列表
  def canopy_4( cluster:(Int,DenseVector[Double]),
                Canopylist:List[(Int,DenseVector[Double])],T2:Double  ):
                List[(Int,(DenseVector[Double],Int) )]={
   val dist = Canopylist.map(
          center => (center._1, norm(center._2 - cluster._2), cluster._1) )
    val index=dist.filter(x=>x._2<T2).map(
              y=>(y._1,(cluster._2,cluster._1) ))  //注意filter也是转化操作，还是返回RDD,但collect之后就不是了
    index // （index，point）List ,index为point对应的Canopy键
  }


  //TODO 这里是main函数
  def main(arg: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()

    //TODO 给sc分发给单机
    val conf = new SparkConf().setAppName("AppName").setMaster("local[3]") //local[4]这里表示分了4个单机
    val sc = new SparkContext(conf)
    //sc.stop()

    //TODO 这里可根据需要修改相对路径或者绝对路径，导入数据
   // var lines = sc.textFile("/Users/hongruoyu/Desktop/iris_noclass_blank.data" ,5) //这里读入的时候就可以控制数据集数量
   //var lines = sc.textFile("/Users/hongruoyu/Desktop/soybean_noclass_blank.data",5)
    var lines = sc.textFile("./src/main/scala/wine_noclass_blank.data",5)
    //注意这里用textFile读取的是未序列化的文件，要用map格式化
    val data = lines.map(line => DenseVector(line.split(",")).map(_.toDouble)).cache()
//    val T1:Double=2 //FIXME
//    val T2:Double=0.8  //FIXME
//    val T3:Double=2

//        val T1:Double=4 //FIXME
//        val T2:Double=2 //FIXME
//        val T3:Double=4.2

            val T1:Double=400 //FIXME
            val T2:Double=200 //FIXME
            val T3:Double=400



    //1 获取局部Canopy中心点列表（center）----------------------------------
    val crudeCenters = data.map {
      x => (canopy_1(x,T1).asInstanceOf[DenseVector[Double]]) //这里是为了判断每个点是否与mapCenters中的中心点距离>T1，否则成为新的中心点
    }.filter(y=>y!=null).collect().toList


    //2 获取全局Canopy中心点列表（index，center）----------------------------------
    val CanopyCenterList=canopy_2(crudeCenters,T3) //这个目前是List(index ,DenseVector) 而非RDD，因为canopy_2函数传出的时候是List
   val canopysize=CanopyCenterList.size //获取canopy数量
    println(s"全局canopy数量为$canopysize")


    //3 划分数据点至Canopy子集,找到每个点对应的Canopy index集合, 处理成（canopy index,point）的形式返回----------------------------------
    val CanopyT2=data.map(x=>
      canopy_3(x,CanopyCenterList,T2)
    ) //返回多个 List[（canopy index 1,point）,（canopy index 2,point）]
    //CanopyT2.take(3)
    val Point_canopy= CanopyT2.flatMap(x=>x) //将列表元素全部展平构成RDD  //从这里开始变成147

 // println(4)


    //4 Canopy_kmeans ----------------------------------

    //4.1 kmeans参数设定
    val convergeDist=0.001 //底数，指数 迭代终止限 //FIXME
    val MaxIteration=20 //kmeans最大迭代次数  //FIXME

    var tempDist=2.0 //当前两次迭代cluster中心点距离差 这里tempDist得大于终止限 convergeDist
    var tempIteration=0 //当前迭代次数
    var ClusterCenterSet= Map(CanopyCenterList:_*) //Map(index ->DenseVector) 用于存储更新cluster中心，并用于最后的输出
      //:_* 操作符用于解包 List 中的元组，将其转化为一系列 key-value 对，然后将其作为参数传递给 Map 的 apply 方法。
    var ClusterCanopy= sc.makeRDD( CanopyCenterList.map{ case (index:Int,vector: DenseVector[Double]) =>(index,(vector,index) )  }  )
                //初始cluster对应的canopy编号即为canopy中心编号，另外还得加上原来的cluster index (canopy index,(cluster,cluster index ))
                //case可用于解构元组元素

      var pc_index1=sc.parallelize(Seq.empty[(Int,DenseVector[Double])])
      //4.2 循环迭代
     //println(tempDist>convergeDist && tempIteration< MaxIteration)
    while(tempDist >convergeDist && tempIteration< MaxIteration ){
     // var tempDist=2.0

      //4.2.1以conpy中心index为键，连接point和cluster
      val combine_pc=Point_canopy.join( ClusterCanopy )  //( canopy index, ( point,(cluster, cluster index)   ) ) 返回RDD

      //4.2.2计算每个点到同canopy cluster的距离
      val pc_dist=combine_pc.map{ case(key,pair) => (pair._1, (norm(pair._1- pair._2._1 ) ,pair._2._2 ) ) }     //pair._2._2 是cluster index
                          //返回的( point ,(dist,cluster index )),此时key转为了 point
      /*
      //这里想一想怎么改，不能和原来那样写，因为一个point对应了多个canopy，得先找到每个point最近的cluster label 才可以 （label,(point._2,1) ）
     // val closest_pc=Point_canopy.map(point => (ClosestCenter( point, ClusterCenterList),(point._2,1)  ) )
      */

      //4.2.3找到每个point的最小cluster点
      val pc_index=pc_dist.reduceByKey{ case( (dist1,index1),(dist2,index2))=> if (dist1<dist2) (dist1,index1) else (dist2,index2)  }
      //返回（key:point, closet dist, cluster index）

      //转化为（cluster index ，point）格式
      //val pc_index1=pc_index.map{ case(point,(dist,cindex) ) => (cindex,point)  }
      pc_index1=pc_index.map{ case(point,(dist,cindex) ) => (cindex,point)  }    //TODO: 数据划分结果是这个变量
      //4.2.4计算新的cluster质心
      val newClusterCenterSet= ( pc_index1
        .map{ //不知道为啥.map 不能写到pc_index1下一行，不然就会有问题 ;如果非要这样写，得在整个代码段首尾加上()
        case(index,point)=> (index, (point, 1) ) }
        .reduceByKey{ (x,y)=> (x._1+y._1, x._2+y._2) }
        //.reduceByKey{ case( (s1,t1),(s2,t2) )=>(s1+s2,t1+t2) }
        .map{ case (index,(s,t))=> (index,s/ DenseVector.fill(s.length)(t.toDouble) ) }).collect().toMap
        // 返回（index，center_vector） 的Map格式，这里必须用collect才能Map

      //4.2.5更新新的cluster中心点列表，并分配其canopy中心

      // 更新新cluster中心点所属的Canopy                   //FIXME 这里也可以调参T2
      val ClusterT2 = newClusterCenterSet.map(x => //这里返回合适的ClusterCanopy,将canopy_3函数略作修改，输入中包含cluster index，返回中添加cluster index
        canopy_4(x, CanopyCenterList, T2) //返回多个 List[（canopy index 1,(point,cluster index)）,（canopy index 2,(point,cluster index)）]
      ).flatMap(x => x).toArray  //这里转成array 为了方便转RDD

      //ClusterCanopy 对应的格式(canopy index,(cluster,cluster index ))
      ClusterCanopy = sc.makeRDD(ClusterT2) //FIXME 这里不知道为啥之前可以makeRDD，现在直接用Listmake不行

      tempDist=0.0 //注意这里要在外面定义var ，里面修改值为0.0 ,不能循环内外同时定义，否则迭代不会停止（虽然不太清楚为什么）
      for( (index,center)<-newClusterCenterSet ){//RDD也可以遍历
       // print(index,center)
        tempDist += squaredDistance(  center, ClusterCenterSet.apply(index) )
        ClusterCenterSet.update(index,center)//更新cluster中心点  //聚类中心点在这
        //newClusterCenterSet.map( x=>x._2) //Map格式 map后第一个元素为key，第二个为value
      }//for

      //println(tempDist)
      tempIteration+=1
//      if(tempDist<convergeDist ) {println("--------------------") }
      println(s"这是第$tempIteration 迭代，这是本次迭代cluster变化大小：$tempDist")
//         if(tempDist>convergeDist && tempIteration< MaxIteration)println("***************")

    }//while

    val endTime = System.currentTimeMillis()
    val totalTime = endTime - startTime
    println("Total runtime: " + totalTime + " ms")
    //pc_index1.foreach(println)
   ClusterCenterSet.foreach(println)

  } //main

} //object














