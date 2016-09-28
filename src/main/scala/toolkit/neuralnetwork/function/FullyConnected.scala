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
import DifferentiableField.GradientPort
import toolkit.neuralnetwork.operator.{backFC, forwardFC, weightGradFC}


class FullyConnected private[FullyConnected] (input: DifferentiableField, weights: DifferentiableField) extends DifferentiableField {
  require(weights.forward.tensorOrder == 1, "weights must be a vector field")
  require(input.forward.fieldShape == weights.forward.fieldShape,
    s"weights must have field shape ${input.forward.fieldShape}, currently ${weights.forward.fieldShape}")
  require(input.forward.tensorShape(0) % input.batchSize == 0,
    s"input vector length (${input.forward.tensorShape(0)}) must be an integer multiple of the batch size (${input.batchSize})")
  private val inputLen = input.forward.tensorShape(0) / input.batchSize
  private val outLen = weights.forward.tensorShape(0) / inputLen
  require(outLen > 0, s"weights have vector length ${weights.forward.tensorShape(0)}, must be at least vector length $inputLen")
  require(weights.forward.tensorShape(0) % outLen == 0,
    s"weight vector length (${weights.forward.tensorShape(0)}) must be a integer multiple of the output length ($outLen)")

  override val inputs: Map[Symbol, GradientPort] = Map(
    'input -> GradientPort(input, inJacobian, inJacobianAdjoint),
    'weights -> GradientPort(weights, weightsJacobian, weightsJacobianAdjoint))
  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = forwardFC(input.forward, weights.forward, batchSize)

  private def inJacobian(dxIn: Field): Field = {
    forwardFC(dxIn, weights.forward, batchSize)
  }

  private def inJacobianAdjoint(grad: Field): Field = {
    backFC(grad, weights.forward, batchSize)
  }

  private def weightsJacobian(dxW: Field): Field = {
    forwardFC(input.forward, dxW, batchSize)
  }

  private def weightsJacobianAdjoint(grad: Field): Field = {
    weightGradFC(input.forward, grad, batchSize)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, weights)
}

/** Factory object- eliminates clutter of 'new' operator. */
object FullyConnected {
  def apply(input: DifferentiableField, weights: DifferentiableField) =
    new FullyConnected(input, weights)
}
