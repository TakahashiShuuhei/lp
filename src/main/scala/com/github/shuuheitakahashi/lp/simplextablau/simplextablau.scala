package com.github.shuuheitakahashi.lp.simplextablau

import com.github.shuuheitakahashi.lp.utils.MatrixCaster._
import com.github.shuuheitakahashi.lp.utils.VectorCaster._
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.slf4j.LoggerFactory

/**
  * Minimize : cx
  * Subject to : Ax = b, x >= 0
  */
class LinearProgramming (val c: Vector,  val A: Matrix,  val b: Vector)

class SimplexTablau(val lp: LinearProgramming) {
  val logger = LoggerFactory.getLogger(this.getClass)

  var topRow: Vector = _
  var table: Array[Vector] = _

  /**
    * lpからテーブルを初期化
    */
  def init(): Unit = {
    val topArr = new Array[Double](lp.c.size + 1)
    lp.c.foreachActive((i, d) => topArr(i) = d)
    topRow = Vectors.dense(topArr)

    table = new Array[Vector](lp.A.numRows)
    for (i <- 0 until lp.A.numRows) {
      val arr = new Array[Double](lp.c.size + 1)
      lp.A.getRow(i).foreachActive((ii, d) => arr(ii) = d)
      arr(arr.length - 1) = lp.b(i)
      table(i) = Vectors.dense(arr)
    }
  }

  def solve(): (Double, Vector) = {
    logger.info("solve start")
    init()

    while (!calcCompleted()) {
      iter()
    }

    val arr = new Array[Double](table.length)
    for (i <- table.indices) {
      arr(i) = table(i)(table(i).size - 1)
    }
    logger.info("solve end")
    (topRow(topRow.size - 1), Vectors.dense(arr))
  }

  def iter(): Unit = {
    val pivotColIdx = choosePivotCol()
    val pivotRowIdx = choosePivotRow(pivotColIdx)
    logger.debug("iter col: {}, row: {}", pivotColIdx, pivotRowIdx)
    sweepOut(pivotColIdx, pivotRowIdx)
  }

  def choosePivotCol(): Int = {
    for (i <- 0 until topRow.size - 1) {
      if (topRow(i) < 0) {
        return i
      }
    }
    throw new IllegalStateException("solveが完了していないのに負の要素がtopRowに存在しない")
  }

  def choosePivotRow(colIdx: Int): Int = {
    var minPair : (Int, Double) = (-1, Double.MaxValue)
    for (i <- table.indices) {
      val row: Vector = table(i)
      val v = row(row.size - 1) / row(colIdx)
      if (v < minPair._2) {
        minPair = (i, v)
      }
    }
    if (minPair._1 < 0) {
      throw new IllegalStateException("pivotRowを選択できない")
    }
    minPair._1
  }

  def sweepOut(colIdx: Int, rowIdx: Int): Unit = {
    val pivotRow  = table(rowIdx) * (1.0 / table(rowIdx)(colIdx))
    table(rowIdx) = pivotRow

    for (i <- table.indices) {
      if (i != rowIdx) {
        val row = table(i)
        table(i) = row - (pivotRow * row(colIdx))
      }
    }

    topRow = topRow - (pivotRow * topRow(colIdx))
  }

  def calcCompleted(): Boolean = {
    var finish = true
    topRow.foreachActive((i, d) => {
      if (i < topRow.size - 1 && d < 0) {
        finish = false
      }
    })
    finish
  }
}