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
import toolkit.neuralnetwork.operator.{spray, sumSpray}


/** The softmax (multinomial logistic regression).  Converts the input to a form that could be
  * considered a discrete probability distribution- i.e. all positive values that sum to 1.
  *
  * @author Dick Carter
  * @param input    The input signal, typically a classification output.
  * @param safeMode Protect against generating NaNs for large inputs (>100).
  */
class Softmax private[Softmax] (input: DifferentiableField, safeMode: Boolean) extends DifferentiableField {
  private val x1 = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(x1)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, jacobian(_, x1), jacobianAdjoint(_, x1)))

  // Perform softmax on a node without any NaN protection
  private def softMax(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in
    val softmax = exp(x) / sumSpray(exp(x), batchSize)
    (softmax, batchSize)
  }

  // Alter the input to guard against NaN's in a subsequent softmax calculation by offsetting
  // all values within an image of the batch so that the maximum value is 0.
  private def shiftDownByMax(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in
    val inputLen = x.tensorShape(0) / batchSize
    // Add protection against generating NaNs where avoidable.
    // For example, if an input is 100, then the softmax will calculate
    // e^100 / (... + e^100 + ...) , which is Infinity / Infinity or NaN.
    //
    // If we shift all values down by the maximum value (i.e. subtract the maximum),
    // then the NaN is avoided without disturbing the value of the calculation otherwise.
    val maxOverBatch = x.blockReduceMax(inputLen)
    val safe = x - spray(maxOverBatch, inputLen)
    (safe, batchSize)
  }

  private def _forward(in: (Field, Int)): (Field, Int) = {
    softMax(if (safeMode) shiftDownByMax(in) else in)
  }

  private def jacobian(dx: Field, in: (Field, Int)): Field = {
    val (softmax, batchSize) = _forward(in)
    softmax * (dx - sumSpray(softmax * dx, batchSize))
  }

  // The jacobian has a symmetric matrix representaion and so is its own transpose.
  // Thus, the jacobianAdjoint equals the jacobian.
  private def jacobianAdjoint(grad: Field, in: (Field, Int)): Field = {
    jacobian(grad, in)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, safeMode)
}

/** Factory object- eliminates clutter of 'new' operator. */
object Softmax {
  /** The softmax (multinomial logistic regression).  Converts the input to a form that could be
    * considered a discrete probability distribution- i.e. all positive values that sum to 1.
    *
    * @param input    The input signal, typically a classification output.
    * @param safeMode Protect against generating NaNs for large inputs (>100).
    */
  def apply(input: DifferentiableField, safeMode: Boolean = true) =
    new Softmax(input, safeMode)
}
