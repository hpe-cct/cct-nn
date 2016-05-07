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

/** Tests the self-consistency of the forward/jacobian/jacobianAdjoint functions of the frequency convolution operator.
  *
  * @author Matthew Pickett
  */
class FrequencyConvolutionSpec extends UnitSpec with ComputeTests {
  val inputShapes = Seq(Shape(32, 64), Shape(5, 5))
  val inputLens = Seq(11, 11 * 13)
  val batchSizes = Seq(59, 1)

  "The frequency convolution op" should "support 2D input" in {
    val node = {
      a: Seq[DifferentiableField] => FrequencyConvolution(a.head, a(1), BorderCyclic)
    }

    jacobian(node, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(node, inputShapes, inputLens, batchSizes)
  }
}