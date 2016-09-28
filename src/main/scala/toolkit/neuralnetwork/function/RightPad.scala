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
import toolkit.neuralnetwork.DifferentiableField.GradientPort

/** Pad an N-dimensional input with zeros on the "right" (after the data in each dimension)
  *
  * @param input the input signal
  * @param sizes the amount of padding to add for all dimensions
  * @author Ben Chandler
  */
class RightPad private[RightPad] (input: DifferentiableField, sizes: Seq[Int]) extends DifferentiableField {
  assert(input.forward.fieldShape.dimensions == 1 || input.forward.fieldShape.dimensions == 2, "input must be 1D or 2D")
  assert(sizes.length == input.forward.fieldShape.dimensions, "pad dimensionality must match input dimensionality")
  assert(sizes.forall(_ >= 0), "pad sizes must be positive or zero")

  private val x = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(x)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, jacobian(_, x), jacobianAdjoint(_, x)))

  private def _forward(x: (Field, Int)): (Field, Int) = {
    assert(x._1.fieldShape.dimensions == 1 || x._1.fieldShape.dimensions == 2, "input must be 1D or 2D")
    assert(sizes.length == x._1.fieldShape.dimensions, "pad dimensionality must match input dimensionality")
    assert(sizes.forall(_ >= 0), "pad sizes must be positive or zero")

    val newShape = Shape(x._1.fieldShape.toArray.zip(sizes).map(p => p._1 + p._2))
    val padded = expand(x._1, BorderZero, newShape)

    (padded, x._2)
  }

  private def jacobian(dx: Field, x: (Field, Int)): Field = _forward((dx, x._2))._1

  private def jacobianAdjoint(grad: Field, x: (Field, Int)): Field =
  // Converting the input shape to a sequence of ranges to grab the middle of
  // the gradient field.
    grad(grad.fieldShape.toArray.zip(sizes).map(s => 0 until s._1 - s._2): _*)

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, sizes)
}

/** Factory object- eliminates clutter of 'new' operator. */
object RightPad {
  /** Pad an N-dimensional input with zeros on the "right" (after the data in each dimension)
    *
    * @param input the input signal
    * @param sizes the amount of padding to add for all dimensions
    */
  def apply(input: DifferentiableField, sizes: Seq[Int]) =
    new RightPad(input, sizes)
}
