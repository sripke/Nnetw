/* Nnetw.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object Nnetw {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Neural Network")
    val sc = new SparkContext(conf)

    // Build a distributed RowMatrix w for weights
    val wRows = sc.textFile("hdfs://localhost:9000/w.txt").map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.sparse(values.length,
                    values.zipWithIndex.map(e => (e._2, e._1)).filter(_._2 != 0.0))
    }
    val w = new RowMatrix(wRows)

    // Create dense vector (matrix) a for input
    val aRows = sc.textFile("hdfs://localhost:9000/a.txt").map(line => line.split(' ').map(_.toDouble)).collect()

    val a = Matrices.dense(3, 2, aRows.transpose.flatten)

    val res = w.multiply(a)
    res.rows.saveAsTextFile("hdfs://localhost:9000/result.txt")
  }
}
