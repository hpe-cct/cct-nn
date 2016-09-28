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

/** Crop an N-dimensional input from the "right" (the highest-indexed data in each dimension).
  *
  * @param input     the input signal
  * @param cropSizes the amount of padding to remove for all dimensions
  * @author Ben Chandler
  */
class RightCrop private[RightCrop] (input: DifferentiableField, cropSizes: Seq[Int]) extends DifferentiableField {
  private val x = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = _forward(x)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => jacobian(dx, x), grad => jacobianAdjoint(grad, x)))

  private def _forward(x: (Field, Int)): (Field, Int) = {
    assert(x._1.fieldShape.dimensions == 1 || x._1.fieldShape.dimensions == 2, "input must be 1D or 2D")
    assert(cropSizes.length == x._1.fieldShape.dimensions, "crop dimensionality must match input dimensionality")
    assert(cropSizes.forall(_ >= 0), "crop sizes must be positive or zero")

    val newShape = x._1.fieldShape.toArray.zip(cropSizes).map(p => 0 until (p._1 - p._2))
    assert(newShape.forall(_.nonEmpty), "crop sizes must be smaller than input field size")
    val cropped = x._1(newShape: _*)

    (cropped, x._2)
  }

  private def jacobian(dx: Field, x: (Field, Int)): Field = _forward((dx, x._2))._1

  private def jacobianAdjoint(grad: Field, x: (Field, Int)): Field =
    expand(grad, BorderZero, Shape(grad.fieldShape.toArray.zip(cropSizes).map(s => s._1 + s._2)))

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, cropSizes)
}

/** Factory object- eliminates clutter of 'new' operator. */
object RightCrop {
  /** Crop an N-dimensional input from the "right" (the highest-indexed data in each dimension).
    *
    * @param input     the input signal
    * @param cropSizes the amount of padding to remove for all dimensions
    */
  def apply(input: DifferentiableField, cropSizes: Seq[Int]) =
    new RightCrop(input, cropSizes)
}
