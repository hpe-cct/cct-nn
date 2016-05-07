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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the bias operator.
  *
  * @author Ben Chandler
  */
class BiasSpec extends UnitSpec with ComputeTests {
  def fn(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    Bias(s.head, s(1))
  }

  def fns(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    Bias(s.head, s(1), sharedBias = true)
  }

  "The bias operator" should "support 0D inputs (unshared)" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs (unshared)" in {
    val inputShapes = Seq(Shape(12), Shape(12))
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D inputs (unshared)" in {
    val inputShapes = Seq(Shape(14, 18), Shape(14, 18))
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 0D inputs (shared)" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fns, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fns, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs (shared)" in {
    val inputShapes = Seq(Shape(12), Shape())
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fns, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fns, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D inputs (shared)" in {
    val inputShapes = Seq(Shape(14, 18), Shape())
    val inputLens = Seq(64, 64)
    val batchSizes = Seq(20, 1)

    jacobian(fns, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fns, inputShapes, inputLens, batchSizes)
  }
}
