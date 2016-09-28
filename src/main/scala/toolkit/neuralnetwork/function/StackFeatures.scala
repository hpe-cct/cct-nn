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

/** Binary compute node that stacks fields in the feature vector domain. Both inputs
  * must have the same field shape.
  *
  * @author Matthew Pickett
  * @param left  first input signal
  * @param right second input signal, stacked on the first in the vector domain
  */
class StackFeatures private[StackFeatures] (left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
  require(left.forward.tensorShape.dimensions == 1, s"'left' must be a VectorField, got ${left.forward.fieldType}")
  require(right.forward.tensorShape.dimensions == 1, s"'right' must be a VectorField, got ${right.forward.fieldType}")
  require(left.forward.tensorShape(0) % left.batchSize == 0,
    s"'left' vector length (${left.forward.tensorShape(0)}) must be an integer multiple of the batch size (${left.batchSize})")
  require(right.forward.tensorShape(0) % right.batchSize == 0,
    s"'right' vector length (${right.forward.tensorShape(0)}) must be an integer multiple of the batch size (${right.batchSize})")
  require(left.forward.fieldShape == right.forward.fieldShape,
    s"'left' and 'right' must have the same field shape, got ${left.forward.fieldShape} and ${right.forward.fieldShape}")
  require(left.batchSize == right.batchSize,
    s"'left' and 'right' must have the same batch size, got ${left.batchSize} and ${right.batchSize}")

  private val x1 = (left.forward, left.batchSize)
  private val x2 = (right.forward, right.batchSize)

  override val batchSize: Int = left.batchSize
  override val forward: Field = _forward(x1, x2)._1
  override val inputs: Map[Symbol, GradientPort] = Map(
    'left -> GradientPort(left, dx => jacobian1(dx, x1, x2), grad => jacobianAdjoint1(grad, x1, x2)),
    'right -> GradientPort(right, dx => jacobian2(dx, x1, x2), grad => jacobianAdjoint2(grad, x1, x2)))

  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    //determine batch size
    val (input1, b1) = x1
    val (input2, b2) = x2
    require(b1 == b2)
    val b = b1

    require(input1.tensorShape.dimensions == 1)
    require(input2.tensorShape.dimensions == 1)
    require(input1.tensorShape(0) % b == 0)
    require(input2.tensorShape(0) % b == 0)
    require(input1.fieldShape == input2.fieldShape)
    val inLen1 = input1.tensorShape(0) / b
    val inLen2 = input2.tensorShape(0) / b
    val outLen = inLen1 + inLen2
    val outType = new FieldType(input1.fieldShape, Shape(outLen * b), Float32)
    val out = GPUOperator(outType, "StackFeatures") {
      _globalThreads(outType.fieldShape, outType.tensorShape)
      val batchIndex = _tensorElement / outLen
      val outIndex = _tensorElement % outLen
      val out = _floatVar()
      _if(outIndex < inLen1) {
        out := _readTensorElement(input1, outIndex + batchIndex * inLen1)
      }
      _else {
        out := _readTensorElement(input2, outIndex - inLen1 + batchIndex * inLen2)
      }
      _writeTensorElement(_out0, out, _tensorElement)
    }
    (out, b)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (input1, b1) = x1
    val zero2 = (x2._1 * 0f, x2._2)

    require(dx1.fieldType == input1.fieldType)
    _forward((dx1, b1), zero2)._1
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (input1, b1) = x1
    val (input2, b2) = x2

    require(b1 == b2)
    val b = b1

    require(input1.tensorShape.dimensions == 1)
    require(input2.tensorShape.dimensions == 1)
    require(grad.tensorShape.dimensions == 1)
    require(input1.tensorShape(0) % b == 0)
    require(input2.tensorShape(0) % b == 0)
    require(grad.tensorShape(0) % b == 0)
    require(input1.fieldShape == input2.fieldShape)
    require(grad.fieldShape == input1.fieldShape)
    val inLen1 = input1.tensorShape(0) / b
    val inLen2 = input2.tensorShape(0) / b
    val gradLen = grad.tensorShape(0) / b
    require(gradLen == inLen1 + inLen2)

    val outType = new FieldType(input1.fieldShape, Shape(inLen1 * b), Float32)
    GPUOperator(outType, "StackFeaturesJacobianAdjoint1") {
      _globalThreads(outType.fieldShape, outType.tensorShape)
      val batchIndex = _tensorElement / inLen1
      val outIndex = _tensorElement % inLen1
      val out = _readTensorElement(grad, outIndex + batchIndex * gradLen)
      _writeTensorElement(_out0, out, _tensorElement)
    }
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (input2, b2) = x2
    val zero1 = (x1._1 * 0f, x1._2)

    require(dx2.fieldType == input2.fieldType)
    _forward(zero1, (dx2, b2))._1
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (input1, b1) = x1
    val (input2, b2) = x2

    require(b1 == b2)
    val b = b1

    //determine in1, in2 and grad len
    require(input1.tensorShape.dimensions == 1)
    require(input2.tensorShape.dimensions == 1)
    require(grad.tensorShape.dimensions == 1)
    require(input1.tensorShape(0) % b == 0)
    require(input2.tensorShape(0) % b == 0)
    require(grad.tensorShape(0) % b == 0)
    require(input1.fieldShape == input2.fieldShape)
    require(grad.fieldShape == input1.fieldShape)
    val inLen1 = input1.tensorShape(0) / b
    val inLen2 = input2.tensorShape(0) / b
    val gradLen = grad.tensorShape(0) / b
    require(gradLen == inLen1 + inLen2)

    val outType = new FieldType(input1.fieldShape, Shape(inLen2 * b), Float32)
    GPUOperator(outType, "StackFeaturesJacobianAdjoint2") {
      _globalThreads(outType.fieldShape, outType.tensorShape)
      val batchIndex = _tensorElement / inLen2
      val outIndex = _tensorElement % inLen2
      val out = _readTensorElement(grad, outIndex + inLen1 + batchIndex * gradLen)
      _writeTensorElement(_out0, out, _tensorElement)
    }
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (left, right)
}

/** Factory object- eliminates clutter of 'new' operator. */
object StackFeatures {
  /** Binary compute node that stacks fields in the feature vector domain. Both inputs
    * must have the same field shape.
    *
    * @param left  first input signal
    * @param right second input signal, stacked on the first in the vector domain
    */
  def apply(left: DifferentiableField, right: DifferentiableField) =
    new StackFeatures(left, right)
}
