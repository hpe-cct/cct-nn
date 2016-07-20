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
import toolkit.neuralnetwork.function.{Bias, TrainableState}
import toolkit.neuralnetwork.policy.{EmptyBinding, LearningRule, WeightInitPolicy, ZeroInit}
import toolkit.neuralnetwork.{DifferentiableField, WeightBinding}


object BiasLayer {
  def apply(input: DifferentiableField, learningRule: LearningRule, sharedBias: Boolean = true,
            initPolicy: WeightInitPolicy = ZeroInit, weightBinding: WeightBinding = EmptyBinding): Layer = {
    val inputShape = input.forward.fieldShape
    val inputTensorShape = input.forward.tensorShape
    require(inputTensorShape.dimensions == 1, s"input must be a vector field, got $inputTensorShape")
    require(inputTensorShape(0) % input.batchSize == 0, s"input vector length (${inputTensorShape(0)}) must be an integer multiple of the batch size (${input.batchSize})")

    val inputLen = inputTensorShape(0) / input.batchSize
    val biasShape = if(sharedBias) Shape() else inputShape
    val weights = TrainableState(biasShape, Shape(inputLen), initPolicy, learningRule, weightBinding)

    Layer(Bias(input, weights, sharedBias), weights)
  }
}
