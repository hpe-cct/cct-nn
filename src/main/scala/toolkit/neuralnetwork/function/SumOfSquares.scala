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


class SumOfSquares private[SumOfSquares] (left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
  private val x1 = (left.forward, left.batchSize)
  private val x2 = (right.forward, right.batchSize)

  override val batchSize: Int = 1
  override val forward: Field = _forward(x1, x2)._1
  override val inputs: Map[Symbol, GradientPort] = Map(
    'left -> GradientPort(left, jacobian1(_, x1, x2), jacobianAdjoint1(_, x1, x2)),
    'right -> GradientPort(right, jacobian2(_, x1, x2), jacobianAdjoint2(_, x1, x2)))

  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val (input1, batchSize1) = x1
    val (input2, batchSize2) = x2
    require(batchSize1 == batchSize2)
    val out = 0.5f * reduceSum(fieldReduceSum(sq(input1 - input2)))
    (out, batchSize1)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val input1 = x1._1
    val input2 = x2._1
    fieldReduceSum(reduceSum((input1 - input2) * dx1))
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val input1 = x1._1
    val input2 = x2._1
    grad * (input1 - input2)
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val input1 = x1._1
    val input2 = x2._1
    fieldReduceSum(reduceSum((input2 - input1) * dx2))
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val input1 = x1._1
    val input2 = x2._1
    grad * (input2 - input1)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (left, right)
}

/** Factory object- eliminates clutter of 'new' operator. */
object SumOfSquares {
  def apply(left: DifferentiableField, right: DifferentiableField) =
    new SumOfSquares(left, right)
}
