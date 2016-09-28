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
import toolkit.neuralnetwork.operator.spray

/** The cross-entropy loss function applied to the softmax of the input relative to
  * the reference signal. This loss function is commonly used for training a classification
  * network.
  *
  * @author Dick Carter
  * @param left  The input signal, typically a classification output
  * @param right The reference signal, typically a one hot code representing a class label
  */
class CrossEntropy private[CrossEntropy] (left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
  private val x1 = (left.forward, left.batchSize)
  private val x2 = (right.forward, right.batchSize)

  override val batchSize: Int = left.batchSize
  override val forward: Field = _forward(x1, x2)._1
  override val inputs: Map[Symbol, GradientPort] = Map(
    'left -> GradientPort(left, dx => jacobian1(dx, x1, x2), grad => jacobianAdjoint1(grad, x1, x2)),
    'right -> GradientPort(right, dx => jacobian2(dx, x1, x2), grad => jacobianAdjoint2(grad, x1, x2)))

  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val (in, batchSize) = x1
    val (ref, batchSize2) = x2
    require(in.fieldType == ref.fieldType, "The field types of both inputs must be equal")
    require(in.fieldShape.dimensions == 0, "Only defined for zero dimensional fields")

    require(batchSize == batchSize2, "The batch sizes of both inputs must be equal")

    val inputLen = in.tensorShape(0) / batchSize
    val crossEntropy = blockReduceSum(ref * -log(in), inputLen)
    (crossEntropy, batchSize)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val (ref, _) = x2
    val inputLen = in.tensorShape(0) / batchSize
    //    blockReduceSum(-(1f/in)*ref*dx1, inputLen)
    blockReduceSum(-ref * dx1 / in, inputLen)
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val (ref, _) = x2
    val inputLen = in.tensorShape(0) / batchSize
    -ref * spray(grad, inputLen) / in
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val inputLen = in.tensorShape(0) / batchSize
    blockReduceSum(dx2 * -log(in), inputLen)
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in, batchSize) = x1
    val inputLen = in.tensorShape(0) / batchSize
    -log(in) * spray(grad, inputLen)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (left, right)
}

/** Factory object- eliminates clutter of 'new' operator. */
object CrossEntropy {
  /** The cross-entropy loss function applied to the softmax of the input relative to
    * the reference signal. This loss function is commonly used for training a classification
    * network.
    *
    * @param left  The input signal, typically a classification output
    * @param right The reference signal, typically a one hot code representing a class label
    */
  def apply (left: DifferentiableField, right: DifferentiableField) =
    new CrossEntropy(left, right)
}