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
import toolkit.neuralnetwork.function.{FullyConnected, TrainableState}
import toolkit.neuralnetwork.policy.{FCInit, LearningRule, StubInit, WeightInitPolicy}
import toolkit.neuralnetwork.DifferentiableField


object FullyConnectedLayer {
  def apply(input: DifferentiableField, outputLen: Int, learningRule: LearningRule, initPolicy: WeightInitPolicy = StubInit): Layer = {
    val inputShape = input.forward.fieldShape
    val inputTensorShape = input.forward.tensorShape
    require(inputTensorShape.dimensions == 1, s"input must be a vector field, got $inputTensorShape")
    require(inputTensorShape(0) % input.batchSize == 0, s"input vector length (${inputTensorShape(0)}) must be an integer multiple of the batch size (${input.batchSize})")
    val inputLen = inputTensorShape(0) / input.batchSize
    require(outputLen >= 1, s"output must contain at least one point, got $outputLen")

    // The FCInit policy requires additional information to scale the weights correctly. If the user specified an
    // initialization policy, use that. Otherwise, replace the default StubInit policy with a populated
    // FCInit instance.
    val realInitPolicy = initPolicy match {
      case StubInit => new FCInit(inputLen)
      case i => i
    }

    val weights = TrainableState(inputShape, Shape(inputLen * outputLen), realInitPolicy, learningRule)
    Layer(FullyConnected(input, weights), weights)
  }
}
