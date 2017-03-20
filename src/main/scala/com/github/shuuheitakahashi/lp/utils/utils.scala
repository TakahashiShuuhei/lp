package com.github.shuuheitakahashi.lp.utils

import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}

class MultipliableVector(val v: Vector) {
  def * (n: Double): Vector = {
    val ar = new Array[Double](v.size)
    v.foreachActive((i, d) => ar(i) = d * n)
    Vectors.dense(ar)
  }
}

class AddableVector(val v: Vector) {
  def + (o: Vector): Vector = {
    if (v.size != o.size) {
      throw new IllegalArgumentException("異なるサイズのVector")
    }
    val ar = new Array[Double](v.size)
    v.foreachActive((i, d) => ar(i) = d)
    o.foreachActive((i, d) => ar(i) = ar(i) + d)
    Vectors.dense(ar)
  }

  def - (o: Vector): Vector = {
    val mv = VectorCaster.vec2MulVec(o)
    this + (mv * -1D)
  }
}

object VectorCaster {
  implicit def vec2MulVec(v: Vector): MultipliableVector = new MultipliableVector(v)
  implicit def vec2AddVec(v: Vector): AddableVector= new AddableVector(v)
}

class AccessibleMatrix(val m: Matrix) {
  def getRow(idx: Int): Vector = {
    val arr = new Array[Double](m.numCols)
    for (i <- 0 until m.numCols) {
      arr(i) = m(idx, i)
    }
    Vectors.dense(arr)
  }
  def getCol(idx: Int): Vector = {
    val arr = new Array[Double](m.numRows)
    for (i <- 0 until m.numRows) {
      arr(i) = m(i, idx)
    }
    Vectors.dense(arr)
  }
}

object MatrixCaster {
  implicit def mat2AccMat(m: Matrix): AccessibleMatrix = new AccessibleMatrix(m)
}
