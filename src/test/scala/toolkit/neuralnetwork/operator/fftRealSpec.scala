/*
 * (c) Copyright 2016 Hewlett Packard Enterprise Development LP
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package toolkit.neuralnetwork.operator

import libcog._
import org.junit.runner.RunWith
import org.scalatest.{Matchers, FunSuite}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class fftRealSpec extends FunSuite with Matchers{
  ignore("FftReal") {
    // To reproduce a failing test with a node seed, create the rng with "new Random(seed)"
    val rng = new Random()
    val rows = math.pow(2, rng.nextInt(9) + 1).toInt
    val columns = math.pow(2, rng.nextInt(9) + 1).toInt
    val vecLen = rng.nextInt(30)

    println(
      s"""Running fftRealSpec with parameters:
         | randomSeed: ${rng.seed}
          | rows: $rows
          | columns: $columns
          | vecLen: $vecLen
     """.stripMargin)

    val cg = new ComputeGraph{
      val
      x = VectorField.random(Shape(rows, columns), Shape(vecLen))
      val fft = fftRI(x)
      val fftR = fft._1
      val fftI = fft._2
      val refHalfSizedR = fftR(0 until rows / 2 + 1, 0 until columns)
      val refHalfSizedI = fftI(0 until rows / 2 + 1, 0 until columns)

      val halfSized = fftReal(x)
      val halfSizedR = halfSized._1
      val halfSizedI = halfSized._2

      val maxDiffR = fieldReduceMax(reduceMax(abs(refHalfSizedR - halfSizedR)))
      val maxDiffI = fieldReduceMax(reduceMax(abs(refHalfSizedI - halfSizedI)))
      val maxDiff = max(maxDiffR, maxDiffI)
      probe(maxDiff)
    }
    cg.step
    val maxDiff = cg.read(cg.maxDiff).asInstanceOf[ScalarFieldReader].read()
    cg.release
    println(s"Maximum difference is: $maxDiff")
    maxDiff should be < 5e-5f
  }
}