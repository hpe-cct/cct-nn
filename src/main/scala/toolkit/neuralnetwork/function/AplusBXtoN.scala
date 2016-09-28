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

/** Function that takes an input X and outputs (A + B*X)^N.  Cog has two pow() function signatures corresponding
  * to both integer and non-integer powers.  The integer case for N is detected here
  * and special-cased (instead of having a separate node for this).
  *
  * If N is other than a positive integer, be aware that you may need to ensure that the A + B*X term
  * is always positive to avoid NaNs from killing the model state.
  *
  * @author Dick Carter
  * @param input the input signal
  * @param a     a constant offset added to the input
  * @param b     a constant multiplier that scales the input
  * @param n     the power to raise the input to
  */
class AplusBXtoN private[AplusBXtoN] (input: DifferentiableField, a: Float, b: Float, n: Float) extends DifferentiableField {
  private val x = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(x)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => jacobian(dx, x), grad => jacobianAdjoint(grad, x)))

  private def isIntPower = n == n.toInt

  private def _forward(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in
    if (isIntPower) {
      n.toInt match {
        case 1 => (a + b * x, batchSize)
        case 0 => throw new IllegalArgumentException("AplusBXtoN node: power 'n' must be non-zero.")
        case intN => (libcog.pow(a + b * x, intN), batchSize)
      }
    }
    else
      (libcog.pow(a + b * x, n), batchSize)
  }

  private def jacobian(dx: Field, in: (Field, Int)): Field = {
    val (x, batchSize) = in
    if (isIntPower) {
      n.toInt match {
        case 1 => b * dx
        case 0 => throw new IllegalArgumentException("AplusBXtoN node: power 'n' must be non-zero.")
        case intN => libcog.pow(a + b * x, intN - 1) * n * b * dx
      }
    }
    else
      libcog.pow(a + b * x, n - 1) * n * b * dx
  }

  private def jacobianAdjoint(grad: Field, in: (Field, Int)): Field = {
    jacobian(grad, in)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, a, b, n)
}

/** Factory object- eliminates clutter of 'new' operator. */
object AplusBXtoN {
  /** Function that takes an input X and outputs (A + B*X)^N.  Cog has two pow() function signatures corresponding
    * to both integer and non-integer powers.  The integer case for N is detected here
    * and special-cased (instead of having a separate node for this).
    *
    * If N is other than a positive integer, be aware that you may need to ensure that the A + B*X term
    * is always positive to avoid NaNs from killing the model state.
    *
    * @param input the input signal
    * @param a     a constant offset added to the input
    * @param b     a constant multiplier that scales the input
    * @param n     the power to raise the input to
    */
  def apply (input: DifferentiableField, a: Float, b: Float, n: Float) =
    new AplusBXtoN(input, a, b, n)
}
