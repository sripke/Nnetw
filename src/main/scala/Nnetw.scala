/* Nnetw.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

class NeuralNet(sizes: List[Int]) {
  val conf = new SparkConf().setAppName("Neural Network")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  val num_layers = sizes.length
  var biases = List[DenseMatrix]()
  var weights = List[RowMatrix]()

  def feedforward(a: DenseMatrix): DenseMatrix {
    return for (e <- biases.zip(weights))
      yield e._2.multiply(a) // + e._1
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
      yield Matrices.randn(e, 1, new java.util.Random).asInstanceOf[DenseMatrix]
    //  yield Matrices.randn(e._1, e._2, new java.util.Random)
    // why not RowMatrix (see below or web) ?? Multiply dist * local vector/matrix(n,1)
    val m = Matrices.randn(2, 3, new java.util.Random).asInstanceOf[DenseMatrix]
    val columns = m.toArray.grouped(m.numRows)
    val vectors = columns.toSeq.map(col => new DenseVector(col.toArray))
    weights = for (e <- sizes.slice(1, num_layers).zip(sizes.slice(0, num_layers - 1)))
      yield new RowMatrix(sc.parallelize(Matrices.randn(2, 3, new java.util.Random).asInstanceOf[DenseMatrix].toArray.grouped(m.numRows).toSeq.map(col => new DenseVector(col.toArray))))
  }

  def read() {
    // Build a distributed RowMatrix w for weights
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
      weights.rows.saveAsTextFile("hdfs://localhost:9000/result.txt")
    }
}

object Test {
  def main(args: Array[String]) {
    val nn = new NeuralNet(args.map(_.toInt).toList)
    nn.randomize()
    nn.print()
  }
}
