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
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.policy.{Best, ConvolutionalLayerPolicy, Freq, Space}

/** Helper function object for creating a convolutional `Compute` `Node` along with its associated `State`
  *
  * @author Matthew Pickett and Dick Carter
  */
object Convolution {
  // TODO: update description
  /** Create `Compute`/`State` `Node` pair.  This selects between a spatial domain and frequency domain
    * implementation depending on the `impl` policy and other parameters.
    *
    * @param input        the input signal DifferentiableField
    * @param weights      the weight DifferentiableField for this convolution
    * @param border       the policy for supplying missing input values for border convolutions
    * @param stride       the downsample factor applied to the output of the convolution
    * @param impl         the policy for selecting the convolution implementation (spacial vs frequency domain)
    * @return The convolutional Node. The associated `ConvolutionalState` node
    *         can be accessed as the `in2` Node member of the returned node.
    */
  def apply(input: DifferentiableField,
            weights: DifferentiableField,
            border: BorderPolicy,
            stride: Int = 1,
            impl: ConvolutionalLayerPolicy = Best): DifferentiableField = {

    impl match {
      case Space =>
        SpatialConvolution(input, weights, border, stride)
      case Freq =>
        FrequencyConvolution(input, weights, border, stride)
      case Best =>
        if (stride > 1)
          SpatialConvolution(input, weights, border, stride)
        else
          FrequencyConvolution(input, weights, border, stride)
    }
  }

  // The FrequencyConvolutional layer's MAC kernels have a strong sensitivity to the underlying GPU architecture
  // (Kepler vs. Maxwell). See the class 'node.compute.FrequencyConvolutional' for use of this flag.
  var tuneForNvidiaMaxwell = false

  // The cct-core compiler can now profile a GPUOperator that has been described as a set of variants.
  // The cct-nn 'MAC' kernels have been coded to take advantage of this.  The first-time compile times
  // will be longer, but after that a cache of previously profiled kernels will result in no overhead
  // to this auto-tuning approach.

  var useProfiler = !Cog.noVariants
}

