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
import toolkit.neuralnetwork.operator.sumSpray


class CrossEntropySoftmax private[CrossEntropySoftmax] (inference: DifferentiableField, labels: DifferentiableField)
  extends DifferentiableField {
  require(inference.batchSize == labels.batchSize,
    s"inference batch size (${inference.batchSize}) must match labels batch size ${labels.batchSize}")
  require(inference.forward.fieldType == labels.forward.fieldType,
    s"inference field type (${inference.forward.fieldType}) must match labels field type ${labels.forward.fieldType}")
  require(inference.forward.fieldShape.dimensions == 0,
    s"only zero-dimensional fields are supported, got ${inference.forward.fieldShape.dimensions}-dimension input")

  private val x1 = (inference.forward, inference.batchSize)
  private val x2 = (labels.forward, labels.batchSize)

  override val inputs: Map[Symbol, GradientPort] = Map(
    'inference -> GradientPort(inference, dx1 => jacobian1(dx1, x1, x2), grad => jacobianAdjoint1(grad, x1, x2)),
    'labels -> GradientPort(labels, dx2 => jacobian2(dx2, x1, x2), grad => jacobianAdjoint2(grad, x1, x2)))
  override val batchSize: Int = 1
  override val forward: libcog.Field = _forward(x1, x2)._1

  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val in = x1._1
    val ref = x2._1
    require(in.fieldType == ref.fieldType, "The field types of both inputs must be equal")
    require(in.fieldShape.dimensions == 0, "Only defined for zero dimensional fields")

    require(x1._2 == x2._2, "The batch sizes of both inputs must be equal")
    val batchSize = x1._2

    val softmax = exp(in) / max(sumSpray(exp(in), batchSize), 1e-4f)
    val logSoftmax = -log(max(softmax, 1e-4f))
    val crossEntropy = reduceSum(fieldReduceSum(ref * logSoftmax))
    (crossEntropy, batchSize)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val in = x1._1
    val ref = x2._1
    val batchSize = x1._2
    val logSoftmaxJacobian = sumSpray(exp(in) * dx1, batchSize) / sumSpray(exp(in), batchSize) - dx1
    reduceSum(fieldReduceSum(ref * logSoftmaxJacobian))
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val in = x1._1
    val ref = x2._1
    val batchSize = x1._2
    sumSpray(ref * grad / sumSpray(exp(in), batchSize), batchSize) * exp(in) - ref * grad
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val in = x1._1
    val batchSize = x1._2
    _forward((in, batchSize), (dx2, batchSize))._1
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val in = x1._1
    val batchSize = x1._2

    val softmax = exp(in) / max(sumSpray(exp(in), batchSize), 1e-4f)
    val logSoftmax = -log(max(softmax, 1e-4f))
    logSoftmax
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (inference, labels)
}

/** Factory object- eliminates clutter of 'new' operator. */
object CrossEntropySoftmax {
  def apply (inference: DifferentiableField, labels: DifferentiableField) =
    new CrossEntropySoftmax(inference, labels)
}
