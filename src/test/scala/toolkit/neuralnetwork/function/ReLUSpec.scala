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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the ReLU operator.
  *
  * @author Matthew Pickett and Ben Chandler
  */
class ReLUSpec extends UnitSpec with ComputeTests {
  val fn = {
    (s: Seq[DifferentiableField]) => ReLU(s.head)
  }

  "The ReLU operator" should "support 0D input" in {
    val inputShapes = Seq(Shape())
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D input" in {
    val inputShapes = Seq(Shape(15))
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D input" in {
    val inputShapes = Seq(Shape(15, 29))
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }
}
