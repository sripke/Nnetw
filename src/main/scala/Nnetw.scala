/* Nnetw.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class NeuralNet(sizes: List[Int]) {

  val conf = new SparkConf().setAppName("Neural Network")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  val num_layers = sizes.length
  var biases = List[DenseVector]()
  var weights = List[RowMatrix]()

  val trainImages = sc.textFile("hdfs://localhost:9000/mnist.tsv").cache
  val data = getLabeledPoints(trainImages)

  def feedforward(a: DenseVector) {
    println("ff biases, weights:") //((3,780)x(780,1)=(3,1), (1,3)x(780,1) <-error
    println(weights(0).numRows())
    println(weights(0).numCols())
    println(weights(1).numRows()) //1
    println(weights(1).numCols()) //3
    println(Matrices.dense(a.size, 1, a.toArray).numRows)//780
    println(Matrices.dense(a.size, 1, a.toArray).numCols)//1
    println("ff biases zip weights:") //(3,780)x(780,1)
    println(biases.zip(weights))
    for (e1 <- biases.zip(weights))
      yield {
        val aMatrix: DenseMatrix = e1._2.multiply(Matrices.dense(a.size, 1, a.toArray))
        a.rows.map(x => Vectors.dense(for(e2 <- x.toArray.zip(e1._1.toArray))
        yield e2._1 + e2._2))
      }
  }
  def sigmoid(z: Double): Double = {
    return 1.0/(1.0 + math.exp(-z))
  }

  def print () {
    println("biases:")
    biases.foreach(println)
    println("weights:")
    weights.foreach(_.rows.foreach(println))
  }

  def randomize() {
    biases = for (e <- sizes.slice(1, num_layers))
      yield Vectors.dense(Matrices.randn(e, 1, new java.util.Random)
        .asInstanceOf[DenseMatrix].toArray).asInstanceOf[DenseVector]

    weights = for (e <- sizes.slice(0, num_layers - 1).zip(sizes.slice(1, num_layers)))
      yield new RowMatrix(sc.parallelize(Matrices.randn(e._1, e._2, new java.util.Random)
        .asInstanceOf[DenseMatrix].toArray.grouped(e._1).toSeq
        .map(col => new DenseVector(col.toArray))))
  }

  def getLabeledPoints(rawData: RDD[String]): RDD[LabeledPoint] = {
    rawData.map { line =>
      val values = line.split("\t").map(_.toDouble)
      LabeledPoint(values.last, Vectors.dense(values.init))
    }
  }
}
/*    // Build a distributed RowMatrix w for weights
    val wRows = sc.textFile("hdfs://localhost:9000/w.txt").map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.dense(values.length,
                    values.zipWithIndex.map(e => (e._2, e._1)).filter(_._2 != 0.0))
    }
    weights = new RowMatrix(wRows)

    // Create dense vector (matrix) a for input
    val aRows = sc.textFile("hdfs://localhost:9000/a.txt").map(line => line.split(' ').map(_.toDouble)).collect()
    val a = Matrices.dense(3, 2, aRows.transpose.flatten)
}
  def write() {
      weights.foreach(_.rows.saveAsTextFile("hdfs://localhost:9000/result.txt"))
    }
}
*/
object Test {
  def main(args: Array[String]) {
    val nn = new NeuralNet(args.map(_.toInt).toList)
    nn.randomize()
    //nn.print()
    nn.feedforward(nn.data.first.features.asInstanceOf[DenseVector])
  }
}
