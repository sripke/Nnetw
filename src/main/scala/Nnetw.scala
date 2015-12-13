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

    val rows = sc.textFile("m1.txt").map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.sparse(values.length,
                    values.zipWithIndex.map(e => (e._2, e._1)).filter(_._2 != 0.0))
    }

    // Create dense vectors for input a and biases b.
    val a: Vector = Vectors.dense(0.1, 0.0, 0.3)
    val b: Vector = Vectors.dense(0.3, 0.2, 0.1)

    // Create a dense matrix w for the weights)
    val w: Matrix = Matrices.dense(3, 2, Array(0.1, 0.3, 0.5, 0.2, 0.4, 0.6))
    //println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
