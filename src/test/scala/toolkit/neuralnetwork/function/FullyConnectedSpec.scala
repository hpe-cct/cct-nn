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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the fully connected operator.
  *
  * @author Matthew Pickett and Ben Chandler
  */
class FullyConnectedSpec extends UnitSpec with ComputeTests {
  val node = {
    a: Seq[DifferentiableField] => FullyConnected(a.head, a(1))
  }

  "The fully-connected operator" should "support 0D input" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(73, 73)
    val batchSizes = Seq(13, 1)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D input" in {
    val inputShapes = Seq(Shape(13), Shape(13))
    val inputLens = Seq(73, 73)
    val batchSizes = Seq(50, 1)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D input" in {
    val inputShapes = Seq(Shape(13, 17), Shape(13, 17))
    val inputLens = Seq(73, 73)
    val batchSizes = Seq(50, 1)

    // Large 2D reductions are less stable than the 0D and 1D cases. Allow this test
    // to pass with a looser tolerance.
    jacobian(node, inputShapes, inputLens, batchSizes, toleranceMultiplier = 2f)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes, toleranceMultiplier = 2f)
  }
}
