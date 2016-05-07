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

package toolkit.neuralnetwork.function

import libcog._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FunSuite, Matchers}
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.source.RandomSource


/** Tests the forward functionality of the MaxPooling operator.
  * See MaxPooling.scala for jacobian/adjointness tests.
  *
  * @author Dick Carter
  */
@RunWith(classOf[JUnitRunner])
class MaxPoolingFunctionalSpec extends FunSuite with Matchers {
  test("MaxPoolingFunctionalSpec") {

    // A parameterized test invoked in multiple ways below.
    def testInstance(inRows: Int, inColumns: Int, vectorLen: Int, poolSize: Int, stride: Int): Unit = {

      val batchSize = 1

      // Helper function to perform max of a bunch of vector fields: max(vf0, vf1, ... )
      def maxFields(fields: Seq[Field]): Field = {
        if (fields.size == 1)
          fields(0)
        else
          max(fields(0), maxFields(fields.drop(1)))
      }
      // Expand a field on the right (and bottom) with the border values.  This is not as
      // simple as one might hope because the CogX core BorderClamp expansion is done
      // cyclically the way one would wish to pad an image for a border clamp convolution using FFT.
      def expandRightBorderClamp(f: Field, newShape: Shape): Field = {
        val addedRows = newShape(0) - f.rows
        val addedColumns = newShape(1) - f.columns
        // Expand by twice the amount, then scale back
        f.expand(BorderClamp, f.rows + 2 * addedRows, f.columns + 2 * addedColumns).trim(newShape)
      }

      // The compute graph for this test instance.  Performs maxPooling using the MaxPooling.forward()
      // operator and also by an alternate approach (the golden answer).
      val cg = new ComputeGraph {
        val inDF = DifferentiableField(VectorField.random(Shape(inRows, inColumns), Shape(vectorLen)) - 0.5f, batchSize)
        val in = inDF.forward
        val outDF = MaxPooling(inDF, poolSize, stride)
        val out = outDF.forward

        val outShape = out.fieldShape

        val inPadded = expandRightBorderClamp(in, Shape(out.rows * stride, out.columns * stride))

        val slices = Array.tabulate(poolSize, poolSize) { (r, c) => inPadded.shift(-r, -c).downsample(stride) }

        val goldenAnswer = maxFields(slices.flatten.toSeq)

        probe(out, goldenAnswer)
      }

      import cg._
      withRelease {
        step
        val outShape = out.fieldShape
        val goldenShape = goldenAnswer.fieldShape
        require(outShape == goldenShape, "Improper output shape for MaxPooling forward operation.")
        val rows = out.rows
        val columns = out.columns
        val actualVector = new Vector(vectorLen)
        val goldenVector = new Vector(vectorLen)

        for (row <- 0 until rows; column <- 0 until columns) {
          read(out).asInstanceOf[VectorFieldReader].read(row, column, actualVector)
          read(goldenAnswer).asInstanceOf[VectorFieldReader].read(row, column, goldenVector)
          require(actualVector == goldenVector)
        }
      }
    }

    testInstance(2, 4, 10, 2, 2)
    testInstance(4, 7, 10, 2, 2)
    testInstance(19, 22, 10, 2, 2)
    testInstance(33, 33, 10, 2, 2)
    testInstance(3, 3, 10, 3, 3)
    testInstance(3, 4, 10, 3, 3)
    testInstance(3, 5, 10, 3, 3)
    testInstance(4, 3, 10, 3, 3)
    testInstance(5, 3, 10, 3, 3)
    testInstance(5, 5, 10, 3, 3)
  }
}
