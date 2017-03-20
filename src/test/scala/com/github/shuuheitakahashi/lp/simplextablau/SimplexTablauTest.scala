package com.github.shuuheitakahashi.lp.simplextablau

import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.scalatest.{FunSuite, ShouldMatchers}


class SimplexTablauTest extends FunSuite with ShouldMatchers {
  test("init") {
    val sut: SimplexTablau = setup

    sut.init()

    assert(sut.topRow === Vectors.dense(-1, -1, 0, 0, 0))
    assert(sut.table.size === 2)
    assert(sut.table(0) === Vectors.dense(3,2,1,0,12))
    assert(sut.table(1) === Vectors.dense(1,2,0,1,8))
  }

  private def setup = {
    val c = Vectors.dense(-1, -1, 0, 0)
    val b = Vectors.dense(12, 8)

    val A = Matrices.dense(2, 4, Array(3, 1, 2, 2, 1, 0, 0, 1))
    val lp = new LinearProgramming(c, A, b)
    val sut = new SimplexTablau(lp)
    sut
  }

  test("calcCompleted") {
    val sut = new SimplexTablau(null)
    sut.topRow = Vectors.dense(1,2,3)
    assert(sut.calcCompleted() === true)

    sut.topRow = Vectors.dense(1,2,-3)
    assert(sut.calcCompleted() === true)

    sut.topRow = Vectors.dense(-1, 2, 3)
    assert(sut.calcCompleted() === false)
  }

  test("choosePivot") {
    val sut = setup
    sut.init()
    val pivotCol = sut.choosePivotCol()
    assert(pivotCol === 0)
    val pivotRow = sut.choosePivotRow(pivotCol)
    assert(pivotRow === 0)
  }

  test("sweepOut") {
    val sut = setup
    sut.init()
    sut.sweepOut(0, 0)

    assertVector(sut.topRow, Vectors.dense(0D, -1D/3, 1D/3, 0D, 4D))
    assertVector(sut.table(0), Vectors.dense(1D, 2D/3, 1D/3, 0D, 4D))
    assertVector(sut.table(1), Vectors.dense(0D, 4D/3, -1D/3, 1D, 4D))
  }

  test("solve") {
    val sut = setup
    val ans = sut.solve()
    assert(ans._1 === 5D)
    assertVector(ans._2, Vectors.dense(2D, 3D))
  }

  test("solve2") {
    // http://www.fujilab.dnj.ynu.ac.jp/lecture/system2.pdf
    // Minimize: z = -400 * x1 - 300 * x2
    // Subject to:
    //    600*x1 + 40*x2 + x3           = 3800
    //    20 *x1 + 30*x2      + x4      = 2100
    //    20 *x1 + 10*x2           + x5 = 1200

    val c = Vectors.dense(-400, -300, 0, 0, 0)
    val A = Matrices.dense(3, 5, Array(600, 20, 20, 40, 30, 10, 1, 0, 0, 0, 1, 0, 0, 0, 1))
    val b = Vectors.dense(3800, 2100, 1200)

    val lp = new LinearProgramming(c, A, b)
    val sut = new SimplexTablau(lp)

    val ans = sut.solve()
    println(ans)
  }

  def assertVector(v1: Vector, v2: Vector): Unit = {
    if (v1.size != v2.size) {
      fail("Vectorのサイズが異なる")
    }
    for (i <- 0 until v1.size) {
      assert(v1(i) === v2(i) +- 0.0001)
    }
  }
}
