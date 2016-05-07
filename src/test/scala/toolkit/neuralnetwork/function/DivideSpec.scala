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
import toolkit.neuralnetwork.{ComputeTests, DifferentiableField, UnitSpec}

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the division operator.
  *
  * @author Matthew Pickett and Ben Chandler
  */
class DivideSpec extends UnitSpec with ComputeTests {
  def fn(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    s.head / s(1)
  }

  "The division operator" should "support 0D inputs" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(11, 11)
    val batchSizes = Seq(59, 59)

    jacobian(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
  }

  it should "support 1D inputs" in {
    val inputShapes = Seq(Shape(37), Shape(37))
    val inputLens = Seq(8, 8)
    val batchSizes = Seq(13, 13)

    jacobian(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
  }

  it should "support 2D inputs" in {
    val inputShapes = Seq(Shape(37,2), Shape(37,2))
    val inputLens = Seq(7, 7)
    val batchSizes = Seq(89, 89)

    jacobian(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
  }

  it should "support 3D inputs" in {
    val inputShapes = Seq(Shape(37,2,7), Shape(37,2,7))
    val inputLens = Seq(9, 9)
    val batchSizes = Seq(12, 12)

    jacobian(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes, positiveXOnly = true)
  }

  it should "throw an exception if the input dimensions don't match" in {
    val inputShapes = Seq(Shape(13, 18), Shape(14, 18))
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 20)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }

  it should "throw an exception if the input lengths don't match" in {
    val inputShapes = Seq(Shape(14, 18), Shape(14, 18))
    val inputLens = Seq(63, 64)
    val batchSizes = Seq(20, 20)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }

  it should "throw an exception if the input batch sizes don't match" in {
    val inputShapes = Seq(Shape(14, 18), Shape(14, 18))
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(21, 20)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    fieldReduceMax(ScalarField(0f))

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }
}
