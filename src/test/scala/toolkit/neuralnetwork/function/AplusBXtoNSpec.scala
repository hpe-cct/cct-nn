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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the AplusBXtoN operator.
  *
  * @author Dick Carter
  */
class AplusBXtoNSpec extends UnitSpec with ComputeTests {
  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  def fn(a: Float, b: Float, n: Float) = {
    args: Seq[DifferentiableField] => AplusBXtoN(args.head, a, b, n)
  }

  "The AplusBXtoNSpec operator" should "support positive integer powers" in {
    val inputShapes = Seq(Shape(24, 22))
    val node = fn(2, 1e-4f, 1)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)

    val node2 = fn(2, 1, 2)

    jacobian(node2, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node2, inputShapes, inputLens, batchSizes)

    val node3 = fn(-2, 4, 3)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support negative integer powers" in {
    val inputShapes = Seq(Shape(24, 22))
    val node = fn(2, 1e-4f, -1)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)

    val node2 = fn(3, 0.1f, -2)

    jacobian(node2, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node2, inputShapes, inputLens, batchSizes)

    val node3 = fn(2, 0.1f, -3)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support positive float powers" in {
    val inputShapes = Seq(Shape(24, 22))
    val node = fn(2, 1e-4f, 0.75f)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)

    val node2 = fn(2, 1e-4f, 1.75f)

    jacobian(node2, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node2, inputShapes, inputLens, batchSizes)

    val node3 = fn(1, 2, 0.5f)

    jacobian(node3, inputShapes, inputLens, batchSizes, positiveXOnly = true)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes, positiveXOnly = true)

  }

  it should "support negative float powers" in {
    val inputShapes = Seq(Shape(24, 22))
    val node = fn(2, 0.01f, -0.5f)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)

    val node2 = fn(2, 1e-4f, -0.75f)

    jacobian(node2, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node2, inputShapes, inputLens, batchSizes)

    val node3 = fn(2, 1e-4f, -2.457f)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "throw an exception with a power of 0" in {
    val inputShapes = Seq(Shape(1, 1))

    val node = fn(2, 1e-4f, 0f)

    an[IllegalArgumentException] shouldBe thrownBy {
      jacobian(node, inputShapes, inputLens, batchSizes)
    }
  }
}
