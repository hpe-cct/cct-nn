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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the MaxPooling operator.
  *
  * @author Matthew Pickett, Ben Chandler and Dick Carter
  */
class MaxPoolingSpec extends UnitSpec with ComputeTests {
  def node(poolSize: Int = 2, stride: Int = 2) = {
    a: Seq[DifferentiableField] => MaxPooling(a.head, poolSize, stride)
  }

  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  "The max pooling operator" should "support even-sized fields" in {
    val inputShapes = Seq(Shape(24, 22))

    jacobian(node(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized rows" in {
    val inputShapes = Seq(Shape(25, 22))

    jacobian(node(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized columns" in {
    val inputShapes = Seq(Shape(24, 23))

    jacobian(node(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized rows and columns" in {
    val inputShapes = Seq(Shape(25, 23))

    jacobian(node(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node(), inputShapes, inputLens, batchSizes)
  }

  it should "generate a warning for 1x1 input" in {
    val inputShapes = Seq(Shape(1, 1))

    an[IllegalArgumentException] shouldBe thrownBy {
      jacobian(node(), inputShapes, inputLens, batchSizes)
    }
  }

  it should "support pooling windows of size 3, stride 3" in {
    val inputShapes = Seq(Shape(24, 21))
    val node3 = node(poolSize = 3, stride = 3)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support pooling windows of size 3, stride 3 requiring input padding" in {
    val inputShapes = Seq(Shape(26, 23))
    val node3 = node(poolSize = 3, stride = 3)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support pooling windows of size 3, stride 2" in {
    val inputShapes = Seq(Shape(25, 23))
    val node3 = node(poolSize = 3, stride = 2)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support pooling windows of size 3, stride 2 requiring input padding" in {
    val inputShapes = Seq(Shape(28, 34))
    val node3 = node(poolSize = 3, stride = 2)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }
}
