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

/** Pull a contiguous region out of a field.
  *
  * @author Dick Carter
  * @param input  the input signal
  * @param ranges the ranges of interest of the input field, e.g. Seq(rowRanges, columnRanges)
  */
class Subspace private[Subspace] (input: DifferentiableField, ranges: Seq[Range]) extends DifferentiableField {
  assert(input.forward.fieldShape.dimensions == 1 || input.forward.fieldShape.dimensions == 2, "input must be 1D or 2D")
  assert(ranges.length == input.forward.fieldShape.dimensions, "ranges dimensionality must match input dimensionality")

  private val x = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(x)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, jacobian(_, x), jacobianAdjoint(_, x)))

  private def _forward(x: (Field, Int)): (Field, Int) = {
    val inField = x._1
    assert(x._1.fieldShape.dimensions == 1 || x._1.fieldShape.dimensions == 2, "input must be 1D or 2D")
    assert(ranges.length == x._1.fieldShape.dimensions, "ranges dimensionality must match input dimensionality")

    val outField = inField(ranges: _*)

    (outField, x._2)
  }

  private def jacobian(dx: Field, x: (Field, Int)): Field = _forward((dx, x._2))._1

  private def jacobianAdjoint(grad: Field, x: (Field, Int)): Field = {
    val inField = x._1
    val origins = ranges.map(r => r.start)
    val ja =
      if (origins.length == 1)
        grad.expand(BorderZero, inField.fieldShape).shiftCyclic(origins(0))
      else if (origins.length == 2)
        grad.expand(BorderZero, inField.fieldShape).shiftCyclic(origins(0), origins(1))
      else
        throw new RuntimeException("ranges dimensionality must match input dimensionality")
    ja
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, ranges)
}

/** Factory object- eliminates clutter of 'new' operator. */
object Subspace {
  /** Pull a contiguous region out of a field.
    *
    * @param input  the input signal
    * @param ranges the ranges of interest of the input field, e.g. Seq(rowRanges, columnRanges)
    */
  def apply(input: DifferentiableField, ranges: Seq[Range]) =
    new Subspace(input, ranges)
}
