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


class ReLU private[ReLU] (input: DifferentiableField) extends DifferentiableField {
  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = max(input.forward, 0f)
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => (forward > 0f) * dx, grad => (forward > 0f) * grad))

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input)
}

/** Factory object- eliminates clutter of 'new' operator. */
object ReLU {
  def apply(input: DifferentiableField) =
    new ReLU(input)
}

