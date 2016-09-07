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


/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the spatial convolution operator.
  *
  * @author Dick Carter
  */
class SpatialConvolutionSpec extends UnitSpec with ComputeTests {
  val inputShapes = Seq(Shape(32, 64), Shape(5, 5))
  val inputLens = Seq(11, 11 * 13)

  def fn(downsampleFactor: Int, borderPolicy: BorderPolicy = BorderValid) = {
    a: Seq[DifferentiableField] => SpatialConvolution(a.head, a(1), stride = downsampleFactor, border = borderPolicy)
  }

  "The spatial conv op (BorderValid)" should "handle batchSize 1" in {
    val batchSizes = Seq(1, 1)
    val downsampleFactor = 1

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize 1 with downsample 2" in {
    val batchSizes = Seq(1, 1)
    val inputLens = Seq(11, 11 * 13)
    val downsampleFactor = 2

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize 1 with downsample 3" in {
    val batchSizes = Seq(1, 1)
    val inputShapes = Seq(Shape(36, 66), Shape(7, 7))
    val inputLens = Seq(12, 12 * 8)
    val downsampleFactor = 3

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1" in {
    val batchSizes = Seq(2, 1)
    val downsampleFactor = 1

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1 with downsample 2" in {
    val batchSizes = Seq(2, 1)
    val inputLens = Seq(11, 11 * 13)
    val downsampleFactor = 2

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1 with downsample 3" in {
    val batchSizes = Seq(10, 1)
    val inputShapes = Seq(Shape(36, 66), Shape(7, 7))
    val inputLens = Seq(4, 4 * 17)
    val downsampleFactor = 3

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle AlexNet CL1-like parameters" in {
    // Like AlexNet layer 1, although with reduced number of filters and small batch size
    val inputRows = 230
    val inputColumns = 230
    val colorPlanes = 3
    val batchSize = 8
    val filterSize = 11
    val numFilters = 8
    val inputShapes = Seq(Shape(inputRows, inputColumns), Shape(filterSize, filterSize))
    val batchSizes = Seq(batchSize, 1)
    val inputLens = Seq(colorPlanes, colorPlanes * numFilters)
    val downsampleFactor = 4

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  "The spatial conv op (BorderZero)" should "handle batchSize 1" in {
    val batchSizes = Seq(1, 1)
    val downsampleFactor = 1

    val node = fn(downsampleFactor, BorderZero)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize 1 with downsample 2" in {
    val batchSizes = Seq(1, 1)
    val inputLens = Seq(11, 11 * 13)
    val downsampleFactor = 2

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize 1 with downsample 3" in {
    val batchSizes = Seq(1, 1)
    val inputShapes = Seq(Shape(36, 66), Shape(7, 7))
    val inputLens = Seq(12, 12 * 8)
    val downsampleFactor = 3

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1" in {
    val batchSizes = Seq(2, 1)
    val downsampleFactor = 1

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1 with downsample 2" in {
    val batchSizes = Seq(2, 1)
    val inputLens = Seq(11, 11 * 13)
    val downsampleFactor = 2

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle batchSize >1 with downsample 3" in {
    val batchSizes = Seq(10, 1)
    val inputShapes = Seq(Shape(36, 66), Shape(7, 7))
    val inputLens = Seq(4, 4 * 17)
    val downsampleFactor = 3

    val node = fn(downsampleFactor)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

  it should "handle AlexNet CL2-like parameters" in {
    // Like AlexNet layer 2, although with reduced number of filters and small batch size
    val inputRows = 55
    val inputColumns = 55
    val inputPlanes = 96
    val batchSize = 8
    val filterSize = 5
    val numFilters = 8
    val inputShapes = Seq(Shape(inputRows, inputColumns), Shape(filterSize, filterSize))
    val batchSizes = Seq(batchSize, 1)
    val inputLens = Seq(inputPlanes, inputPlanes * numFilters)

    val node = fn(1, BorderZero)

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }

}
