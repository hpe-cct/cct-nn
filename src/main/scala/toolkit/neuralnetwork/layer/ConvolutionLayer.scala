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

package toolkit.neuralnetwork.layer

import libcog._
import toolkit.neuralnetwork.function.{Convolution, TrainableState}
import toolkit.neuralnetwork.policy._
import toolkit.neuralnetwork.{DifferentiableField, WeightBinding}


object ConvolutionLayer {
  def apply(input: DifferentiableField,
            filterShape: Shape,
            filterNum: Int,
            border: BorderPolicy,
            learningRule: LearningRule,
            stride: Int = 1,
            impl: ConvolutionalLayerPolicy = Best,
            initPolicy: WeightInitPolicy = ConvInit,
            weightBinding: WeightBinding = EmptyBinding): Layer = {

    val inputShape = input.forward.fieldShape
    val inputTensorShape = input.forward.tensorShape
    require(inputTensorShape.dimensions == 1, s"input must be a vector field, got $inputTensorShape")
    require(inputTensorShape(0) % input.batchSize == 0, s"input vector length (${inputTensorShape(0)}) must be an integer multiple of the batch size (${input.batchSize})")
    require(filterShape.dimensions == 2, s"filters must be 2D, got $filterShape")
    require(filterNum >= 1, s"filter bank must contain at least one filter, got $filterNum")

    val inputLen = inputTensorShape(0) / input.batchSize
    // Allocating `filterNum` filters, each of a shape specified by `filterShape` and `inputLen` planes deep
    val weights = TrainableState(filterShape, Shape(inputLen * filterNum), initPolicy, learningRule, weightBinding)
    Layer(Convolution(input, weights, border, stride, impl), weights)
  }
}
