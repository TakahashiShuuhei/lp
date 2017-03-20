package com.github.shuuheitakahashi.lp.utils

import com.github.shuuheitakahashi.lp.utils.MatrixCaster._
import com.github.shuuheitakahashi.lp.utils.VectorCaster._
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.scalatest.{FunSuite, ShouldMatchers}

class UtilsTest extends FunSuite with ShouldMatchers {
  test("mul") {
    val v1 = Vectors.dense(1.0, 2.0, 3.0)
    val v2 = v1 * 3
    assert(v2 === Vectors.dense(3.0, 6.0, 9.0))
  }

  test("add") {
    val v1 = Vectors.dense(1.0, 2.0, 3.0)
    val v2 = Vectors.dense(2.0, 3.0, 4.0)
    assert(v1 + v2 === Vectors.dense(3.0, 5.0, 7.0))
  }

  test("sub") {
    val v1 = Vectors.dense(1.0, 2.0, 3.0)
    val v2 = Vectors.dense(2.0, 3.0, 4.0)
    assert(v1 - v2 === Vectors.dense(-1.0, -1.0, -1.0))
  }

  test("getRow") {
    val m = Matrices.dense(2, 3, Array(1,2,3,4,5,6))
    assert(m.getRow(0) === Vectors.dense(1,3,5))
  }
}
