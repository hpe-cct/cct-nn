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


class CrossEntropySoftmaxSpec extends UnitSpec with ComputeTests {
  //override def dx = 0.1f

  def fn(s: Seq[DifferentiableField]) = {
    require(s.length == 2)
    CrossEntropySoftmax(s.head, s(1))
  }

  "The cross-entropy softmax operator" should "support batch size 1" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(10, 10)
    val batchSizes = Seq(1, 1)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support typical MNIST input lengths and batch sizes" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(10, 10)
    val batchSizes = Seq(120, 120)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  ignore should "support typical CIFAR input lengths and batch sizes" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(10, 10)
    val batchSizes = Seq(1024, 1024)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  ignore should "support typical AlexNet input lengths and batch sizes" in {
    val inputShapes = Seq(Shape(), Shape())
    val inputLens = Seq(1000, 1000)
    val batchSizes = Seq(128, 128)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "throw an exception for 1D input" in {
    val inputShapes = Seq(Shape(10), Shape(10))
    val inputLens = Seq(10, 10)
    val batchSizes = Seq(1024, 1024)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }

  it should "throw an exception for 2D input" in {
    val inputShapes = Seq(Shape(10, 10), Shape(10, 10))
    val inputLens = Seq(10, 10)
    val batchSizes = Seq(1024, 1024)

    an[IllegalArgumentException] should be thrownBy {
      jacobian(fn, inputShapes, inputLens, batchSizes)
    }

    an[IllegalArgumentException] should be thrownBy {
      jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
    }
  }
}
