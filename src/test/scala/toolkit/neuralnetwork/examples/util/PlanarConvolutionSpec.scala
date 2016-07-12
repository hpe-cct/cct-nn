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

package toolkit.neuralnetwork.examples.util

import libcog._
import toolkit.neuralnetwork.{ComputeTests, DifferentiableField, UnitSpec}


class PlanarConvolutionSpec extends UnitSpec with ComputeTests {
  def fn(borderPolicy: BorderPolicy, samplingPolicy: ConvolutionSamplingPolicy, size: Int = 3) = {
    a: Seq[DifferentiableField] =>
      PlanarConvolution(a.head, ScalarField(size, size, (_, _) => 1f),
        borderPolicy, samplingPolicy)
  }

  val inputLens = Seq(31)
  val batchSizes = Seq(7)
  val inputShapes = Seq(Shape(24, 22))

  "The PlanerConvolution operator" should "support BorderZero/NoSampling policies" in {
    jacobian(fn(BorderZero, NoSamplingConvolution), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(BorderZero, NoSamplingConvolution), inputShapes, inputLens, batchSizes)
  }

  it should "support BorderZero/DownsampleOutput policies" in {
    jacobian(fn(BorderZero, DownsampleOutputConvolution(2)), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(BorderZero, DownsampleOutputConvolution(2)), inputShapes, inputLens, batchSizes)
  }

  it should "support BorderValid/NoSampling policies" in {
    jacobian(fn(BorderValid, NoSamplingConvolution), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(BorderValid, NoSamplingConvolution), inputShapes, inputLens, batchSizes)
  }

  it should "support BorderValid/DownsampleOutput policies" in {
    jacobian(fn(BorderValid, DownsampleOutputConvolution(2)), inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn(BorderValid, DownsampleOutputConvolution(2)), inputShapes, inputLens, batchSizes)
  }
}
