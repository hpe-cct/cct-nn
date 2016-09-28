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

/** Pad an input with zeros.
  *
  * @author Matthew Pickett
  * @param input the input signal
  * @param size  pad size, in field points
  */
class ZeroPad private[ZeroPad] (input: DifferentiableField, size: Int) extends DifferentiableField {
  require(input.forward.fieldShape.dimensions == 1 || input.forward.fieldShape.dimensions == 2, "input must be 1D or 2D")
  require(size > 0, "pad size must be positive")

  private val in = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(in)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => jacobian(dx, in), grad => jacobianAdjoint(grad, in)))

  private def _forward(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in

    val newShape = x.fieldShape.map(_ + size * 2)
    val padded = expand(x, BorderZero, newShape)
    val shifted = x.fieldShape.dimensions match {
      case 1 => shiftCyclic(padded, size)
      case 2 => shiftCyclic(padded, size, size)
    }

    (shifted, batchSize)
  }

  private def jacobian(dx: Field, in: (Field, Int)): Field = {
    val (x, batchSize) = in
    _forward((dx, batchSize))._1
  }

  private def jacobianAdjoint(grad: Field, x: (Field, Int)): Field =
  // Converting the input shape to a sequence of ranges to grab the middle of
  // the gradient field.
    grad(grad.fieldShape.toArray.map(s => size until s - size): _*)

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, size)
}

/** Factory method- eliminates clutter of 'new' operator. */
object ZeroPad {
  def apply (input: DifferentiableField, size: Int) =
    new ZeroPad(input, size)
}
