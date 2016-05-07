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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the Pow operator.
  *
  * @author Dick Carter
  */
class SubspaceSpec extends UnitSpec with ComputeTests {
  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  def fn(ranges: Seq[Range]) = {
    a: Seq[DifferentiableField] => Subspace(a.head, ranges)
  }

  "The subspace operator" should "work with range origins == 0" in {
    val ranges = Seq(0 until 20, 0 until 19)
    val node = fn(ranges)
    val inputShapes = Seq(Shape(27, 30))

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "work with range origins != 0" in {
    val ranges = Seq(3 until 15, 5 until 18)
    val node = fn(ranges)
    val inputShapes = Seq(Shape(24, 22))

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }
}
