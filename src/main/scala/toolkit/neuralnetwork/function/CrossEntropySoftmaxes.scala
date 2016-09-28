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

/** The cross-entropy loss function applied to the softmax of the input relative to
  * the reference signal. This loss function is commonly used for training a classification
  * network.  Unlike the similarly-named "CrossEntropySoftMax", this class computes
  * a cross-entropy softmax individually for each image representation of the batch.  As
  * such, its output is not a single scalar, but instead a vector of length `batchSize`.
  * This allows the class to be tested by the existing test infrastructure.
  *
  * @author Dick Carter
  * @param left          The input signal, typically a classification output
  * @param right         The reference signal, typically a one hot code representing a class label
  * @param refInputIsPDF The `right` reference input for each element of the batch sums to 1.
  * @param safeMode      Protect against generating NaNs for large inputs (>100).
  *
  */
class CrossEntropySoftmaxes private[CrossEntropySoftmaxes] (left: DifferentiableField, right: DifferentiableField,
                                 refInputIsPDF: Boolean, safeMode: Boolean) extends DifferentiableField {
  private val x1 = (left.forward, left.batchSize)
  private val x2 = (right.forward, right.batchSize)

  override val batchSize: Int = left.batchSize
  override val forward: Field = _forward(x1, x2)._1
  override val inputs: Map[Symbol, GradientPort] = Map(
    'left -> GradientPort(left, jacobian1(_, x1, x2), jacobianAdjoint1(_, x1, x2)),
    'right -> GradientPort(right, jacobian2(_, x1, x2), jacobianAdjoint2(_, x1, x2)))

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

  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val (in, batchSize) = x1
    val (ref, batchSize2) = x2
    require(in.fieldType == ref.fieldType, "The field types of both inputs must be equal")
    require(in.fieldShape.dimensions == 0, "Only defined for zero dimensional fields")

    require(batchSize == batchSize2, "The batch sizes of both inputs must be equal")
    val inputLen = in.tensorShape(0) / batchSize

    val softmax = softMax(if (safeMode) shiftDownByMax(x1) else x1)._1
    val logSoftmax = -log(softmax)
    val crossEntropy = blockReduceSum(ref * logSoftmax, inputLen)
    (crossEntropy, batchSize)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val (ref, _) = x2
    val inputLen = in.tensorShape(0) / batchSize

    // This summing of the ref inputs for each element of the batch may not be necessary
    // if the ref input is a one-hot code or some other probability distribution function.
    lazy val refSum = sumSpray(ref, batchSize)

    val softmax = softMax(if (safeMode) shiftDownByMax(x1) else x1)._1
    if (refInputIsPDF)
      blockReduceSum((softmax - ref) * dx1, inputLen)
    else
      blockReduceSum((softmax * refSum - ref) * dx1, inputLen)
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val (ref, _) = x2
    val gradSprayFactor = in.tensorShape(0) / grad.tensorShape.points

    lazy val refSum = sumSpray(ref, batchSize)

    val softmax = softMax(if (safeMode) shiftDownByMax(x1) else x1)._1
    if (refInputIsPDF)
      (softmax - ref) * spray(grad, gradSprayFactor)
    else
      (softmax * refSum - ref) * spray(grad, gradSprayFactor)
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    _forward((in, batchSize), (dx2, batchSize))._1
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val (ref, _) = x2
    val gradSprayFactor = in.tensorShape(0) / grad.tensorShape.points

    val softmax = softMax(if (safeMode) shiftDownByMax(x1) else x1)._1
    -log(softmax) * spray(grad, gradSprayFactor)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (left, right, refInputIsPDF, safeMode)
}

/** Factory object- eliminates clutter of 'new' operator. */
object CrossEntropySoftmaxes {
  /** The cross-entropy loss function applied to the softmax of the input relative to
    * the reference signal. This loss function is commonly used for training a classification
    * network.  Unlike the similarly-named "CrossEntropySoftMax", this class computes
    * a cross-entropy softmax individually for each image representation of the batch.  As
    * such, its output is not a single scalar, but instead a vector of length `batchSize`.
    * This allows the class to be tested by the existing test infrastructure.
    *
    * @param left          The input signal, typically a classification output
    * @param right         The reference signal, typically a one hot code representing a class label
    * @param refInputIsPDF The `right` reference input for each element of the batch sums to 1.
    * @param safeMode      Protect against generating NaNs for large inputs (>100).
    *
    */
  def apply (left: DifferentiableField, right: DifferentiableField,
             refInputIsPDF: Boolean = true, safeMode: Boolean = true) =
    new CrossEntropySoftmaxes(left, right, refInputIsPDF, safeMode)
}
