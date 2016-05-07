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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the RightCrop operator.
  *
  * @author Dick Carter
  */
class RightCropSpec extends UnitSpec with ComputeTests {
  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  def fn(cropSizes: Seq[Int]) = {
    a: Seq[DifferentiableField] => RightCrop(a.head, cropSizes)
  }

  "The RightCrop operator" should "support symmetric cropping" in {
    val inputShapes = Seq(Shape(24, 22))
    val node = fn(Seq(5, 5))

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "support asymmetric cropping" in {
    val inputShapes = Seq(Shape(23, 32))
    val node = fn(Seq(6, 14))

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs" in {
    val inputShapes = Seq(Shape(45))
    val node = fn(Seq(7))

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }
}
