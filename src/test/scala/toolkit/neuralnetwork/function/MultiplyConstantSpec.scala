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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the multiply-constant operator.
  *
  * @author Dick Carter
  */
class MultiplyConstantSpec extends UnitSpec with ComputeTests {
  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  def closure(scale: Float) = {
    (s: Seq[DifferentiableField]) => s.head * scale
  }

  "The multiply-constant operator" should "support positive scales" in {
    val inputShapes = Seq(Shape(24, 22))
    def fn = closure(22f/7)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support negative scales" in {
    val inputShapes = Seq(Shape(24, 22))
    def fn = closure(-12.345f)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support zero scales" in {
    val inputShapes = Seq(Shape(24, 22))
    def fn = closure(0f)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support infix notation" in {
    val inputShapes = Seq(Shape(24, 22))
    def fn(s: Seq[DifferentiableField]) = s.head * 2.345f

    // Not sure this is still possible
    //require(node.isInstanceOf[Scale], "Expecting Scale class, found " + node.getClass())

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }
}
