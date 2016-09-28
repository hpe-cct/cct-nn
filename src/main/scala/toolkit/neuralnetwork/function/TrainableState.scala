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
import toolkit.neuralnetwork.{DifferentiableField, WeightBinding}
import toolkit.neuralnetwork.policy.{LearningRule, WeightInitPolicy}

class TrainableState private[TrainableState] (fieldShape: Shape, tensorShape: Shape, initPolicy: WeightInitPolicy,
                          learningRule: LearningRule, weightBinding: WeightBinding) extends DifferentiableField {
  override val batchSize = 1
  override val gradientConsumer = learningRule.gradientConsumer
  override val forward: Field = {
    // If the weight binding has a stored set of weights, use those. If not,
    // build a new set of weights using the initPolicy and register the resulting
    // field so the WeightBinding can read it later.
    weightBinding.initialWeights.getOrElse {
      val initState = initPolicy.initState(fieldShape, tensorShape)
      weightBinding.register(initState)
      initState
    }
  }

  override def backwardCallback(backward: Field): Unit = {
    learningRule.learn(forward, backward)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (fieldShape, tensorShape, initPolicy, learningRule, weightBinding)
}

/** Factory method- eliminates clutter of 'new' operator. */
object TrainableState {
  def apply (fieldShape: Shape, tensorShape: Shape, initPolicy: WeightInitPolicy,
             learningRule: LearningRule, weightBinding: WeightBinding) =
    new TrainableState(fieldShape, tensorShape, initPolicy, learningRule, weightBinding)
}
