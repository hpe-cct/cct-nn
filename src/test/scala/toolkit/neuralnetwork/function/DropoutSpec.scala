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


class DropoutSpec extends UnitSpec with ComputeTests {
  val inputLens = Seq(31)
  val batchSizes = Seq(7)

  val _seed = Some(Random.nextSeed)

  val fn = {
    (s: Seq[DifferentiableField]) => Dropout(s.head, seed = _seed)
  }

  val fnOff = {
    (s: Seq[DifferentiableField]) => Dropout(s.head, enabled = false, seed = _seed)
  }

  "The dropout operator" should "support 0D inputs when enabled" in {
    val inputShapes = Seq(Shape())

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs when enabled" in {
    val inputShapes = Seq(Shape(12))

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D inputs when enabled" in {
    val inputShapes = Seq(Shape(14, 18))

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 0D inputs when disabled" in {
    val inputShapes = Seq(Shape())

    jacobian(fnOff, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fnOff, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D inputs when disabled" in {
    val inputShapes = Seq(Shape(12))

    jacobian(fnOff, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fnOff, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D inputs when disabled" in {
    val inputShapes = Seq(Shape(14, 18))

    jacobian(fnOff, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fnOff, inputShapes, inputLens, batchSizes)
  }
}
