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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the stack features operator.
  *
  * @author Matthew Pickett and Ben Chandler
  */
class StackFeaturesSpec extends UnitSpec with ComputeTests {
  def fn(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    StackFeatures(s.head, s(1))
  }

  "The stack features operator" should "support 0D inputs" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(64, 32)
    val batchSizes = Seq(12, 12)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs" in {
    val inputShapes = Seq(Shape(13), Shape(13))
    val inputLens = Seq(64, 32)
    val batchSizes = Seq(12, 12)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D inputs" in {
    val inputShapes = Seq(Shape(13, 17), Shape(13, 17))
    val inputLens = Seq(64, 32)
    val batchSizes = Seq(12, 12)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "throw an exception if the input dimensions don't match" in {
    val inputShapes = Seq(Shape(13, 17), Shape(13, 18))
    val inputLens = Seq(64, 32)
    val batchSizes = Seq(12, 12)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }

  it should "throw an exception if the input batch sizes don't match" in {
    val inputShapes = Seq(Shape(13, 17), Shape(13, 17))
    val inputLens = Seq(64, 32)
    val batchSizes = Seq(12, 15)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }
}
