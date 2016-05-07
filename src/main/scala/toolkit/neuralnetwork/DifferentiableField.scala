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
import toolkit.neuralnetwork.DifferentiableField.{GradientBinding, GradientPort}


trait DifferentiableField extends DifferentiableFieldOps with GradientPropagation {
  // A forward field and batch size are mandatory for all differentiable fields
  val forward: Field
  val batchSize: Int
  // All inputs which can carry a gradient must be defined here.
  val inputs: Map[Symbol, GradientPort] = Map()
  // Learned state DFields are gradient consumers. All others are either gradient senders or unused.
  val gradientConsumer: Boolean = false
  // Callback function invoked when the backward field is initialized
  def backwardCallback(back: Field): Unit = {}
  // Mutable state necessary to wire up the gradient.
  var backward: Option[Field] = None
  var gradientBinding: Option[GradientBinding] = None
  // This is a hidden cache field. Call the totalDerivative() method to get the forward gradient.
  private[neuralnetwork] var forwardGradient: Option[Field] = None
}

object DifferentiableField {
  // As a participant in the gradient, a node can be Unused, a Leaf (meaning actively learning state), or
  // a Sender (which passes a gradient through to one or more leaf nodes).
  sealed trait GradientType
  case class Leaf(senders: Set[(DifferentiableField, Symbol)]) extends GradientType
  case class Sender(senders: Set[(DifferentiableField, Symbol)]) extends GradientType
  // A gradient port is an input to the DifferentiableField which can carry a gradient.
  case class GradientPort(df: DifferentiableField, jacobian: Field => Field, jacobianAdjoint: Field => Field)
  // When bound, DifferentiableFields have an owner and gradient type
  case class GradientBinding(owner: DifferentiableField, gradientType: GradientType)

  // Lift a standard field to a DifferentiableField. Lifted fields cannot carry or consume gradients.
  def apply(field: Field, batchSize: Int): DifferentiableField = {
    require(field.tensorOrder == 1, s"tensor order must be exactly one, got ${field.tensorOrder}")
    val _batchSize = batchSize

    new DifferentiableField {
      override val batchSize: Int = _batchSize
      override val forward: libcog.Field = field
    }
  }
}
