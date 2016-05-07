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

package toolkit.neuralnetwork

import libcog._


class DifferentiableFieldSpec extends UnitSpec with ComputeTests {
  "A DifferentiableField" should "support signal splitting" in {
    val inputShapes = Seq(Shape(24, 22))
    val inputLens = Seq(31)
    val batchSizes = Seq(7)

    def fn(input: Seq[DifferentiableField]): DifferentiableField = {
      require(input.length == 1)

      // Split the input, then re-join
      val left = input.head * 3f
      val right = function.ReLU(input.head)
      left + right
    }

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }
}
