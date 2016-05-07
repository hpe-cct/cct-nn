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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the downsample operator.
  *
  * @author Dick Carter
  */
class DownsampleSpec extends UnitSpec with ComputeTests {
  def fn(factor: Int = 2) = {
    a: Seq[DifferentiableField] => Downsample(a.head, factor)
  }

  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  "The downsample operator" should "support even-sized fields" in {
    val inputShapes = Seq(Shape(24, 22))

    jacobian(fn(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized rows" in {
    val inputShapes = Seq(Shape(25, 22))

    jacobian(fn(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized columns" in {
    val inputShapes = Seq(Shape(24, 23))

    jacobian(fn(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(), inputShapes, inputLens, batchSizes)
  }

  it should "support odd-sized rows and columns" in {
    val inputShapes = Seq(Shape(25, 23))

    jacobian(fn(), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(), inputShapes, inputLens, batchSizes)
  }

  it should "support factors of 3" in {
    val inputShapes = Seq(Shape(24, 21))
    val node3 = fn(factor = 3)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support factors of 3 with non multiple-of-3 inputs" in {
    val inputShapes = Seq(Shape(26, 23))
    val node3 = fn(factor = 3)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support factors of 4" in {
    val inputShapes = Seq(Shape(24, 20))
    val node3 = fn(factor = 4)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }

  it should "support factors of 4 with non multiple-of-4 inputs" in {
    val inputShapes = Seq(Shape(26, 23))
    val node3 = fn(factor = 4)
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    jacobian(node3, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node3, inputShapes, inputLens, batchSizes)
  }
}
