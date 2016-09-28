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


/** A bias transformation that adds bias weights to the input signal
  *
  * @author Matthew Pickett
  * @param input      The input signal
  * @param weights    The bias weights
  * @param sharedBias Flag for sharing bias across the input field. Sharing bias causes the
  *                   bias to be applied uniformly across the field points of the input and
  *                   requires that `weights` has a 0D field shape. Unshared bias applies a different
  *                   bias to each field point and requires `weights` to have the same field shape as `input`.
  */
class Bias private[Bias] (input: DifferentiableField, weights: DifferentiableField, sharedBias: Boolean) extends DifferentiableField {
  private val inField = input.forward
  private val inBatchSize = input.batchSize
  private val inputShape = inField.fieldShape
  private val inputTensorShape = inField.tensorShape

  require(inputTensorShape.dimensions == 1,
    s"input tensor shape must be 1-dimensional, got $inputTensorShape")
  require(inputTensorShape(0) % inBatchSize == 0,
    s"input vector length ($inputTensorShape) must be an integer multiple of the batch size ($inBatchSize)")

  require(weights.forward.tensorShape.dimensions == 1,
    s"weight tensor shape must be 1-dimensional, got ${weights.forward.tensorShape}")

  if (sharedBias) require(weights.forward.fieldShape == Shape(),
    s"shared bias must have field shape Shape( ), got ${weights.forward.fieldShape}")
  else require(weights.forward.fieldShape == input.forward.fieldShape,
    s"non-shared bias requires the weight field shape (${weights.forward.fieldShape}) match the input field shape ($inputShape)")

  private val inputLen = inputTensorShape(0) / inBatchSize
  private val biasLen = weights.forward.tensorShape(0)
  require(inputLen == biasLen, s"weight vector length ($biasLen) must match the number of input planes ($inputLen)")


  private val x1 = (input.forward, input.batchSize)
  private val x2 = (weights.forward, weights.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = _forward((input.forward, batchSize), (weights.forward, weights.batchSize))._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => jacobian1(dx, x1, x2), grad => jacobianAdjoint1(grad, x1, x2)),
      'weights -> GradientPort(weights, dx => jacobian2(dx, x1, x2), grad => jacobianAdjoint2(grad, x1, x2)))


  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = x1
    val (b, biasBatchSize) = x2
    require(biasBatchSize == 1)
    require(x.tensorShape(0) % batchSize == 0)
    val inputLen = x.tensorShape(0) / batchSize
    val biasLen = b.tensorShape(0)
    require(inputLen == biasLen)

    if (sharedBias) require(b.fieldShape == Shape())
    else require(b.fieldShape == x.fieldShape)

    val out = GPUOperator(x.fieldType, "biasFwd") {
      _globalThreads(x.fieldShape, x.tensorShape)

      val xElem = x.fieldShape.dimensions match {
        case 0 => _readTensorElement(x, _tensorElement)
        case 1 => _readTensorElement(x, _column, _tensorElement)
        case 2 => _readTensorElement(x, _row, _column, _tensorElement)
        case 3 => _readTensorElement(x, _layer, _row, _column, _tensorElement)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }

      val bElem = b.fieldShape.dimensions match {
        case 0 => _readTensorElement(b, _tensorElement % inputLen)
        case 1 => _readTensorElement(b, _column, _tensorElement % inputLen)
        case 2 => _readTensorElement(b, _row, _column, _tensorElement % inputLen)
        case 3 => _readTensorElement(b, _layer, _row, _column, _tensorElement % inputLen)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }

      x.fieldShape.dimensions match {
        case 0 => _writeTensorElement(_out0, xElem + bElem, _tensorElement)
        case 1 => _writeTensorElement(_out0, xElem + bElem, _column, _tensorElement)
        case 2 => _writeTensorElement(_out0, xElem + bElem, _row, _column, _tensorElement)
        case 3 => _writeTensorElement(_out0, xElem + bElem, _layer, _row, _column, _tensorElement)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }
    }
    (out, batchSize)
  }

  //x + b has an identity jacobian w.r.t. x
  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val x = x1._1
    require(dx1.fieldType == x.fieldType)
    dx1
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val x = x1._1
    require(grad.fieldType == x.fieldType)
    grad
  }

  //x + b has an identity jacobian w.r.t. b
  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (x, batchSize) = x1
    val (b, biasBatchSize) = x2
    require(biasBatchSize == 1)
    require(dx2.fieldType == b.fieldType)
    require(x.tensorShape(0) % batchSize == 0)
    val inputLen = x.tensorShape(0) / batchSize
    val biasLen = b.tensorShape(0)
    require(inputLen == biasLen)

    val out = GPUOperator(x.fieldType, "biasJacobian2") {
      _globalThreads(x.fieldShape, x.tensorShape)

      val bElem = b.fieldShape.dimensions match {
        case 0 => _readTensorElement(dx2, _tensorElement % inputLen)
        case 1 => _readTensorElement(dx2, _column, _tensorElement % inputLen)
        case 2 => _readTensorElement(dx2, _row, _column, _tensorElement % inputLen)
        case 3 => _readTensorElement(dx2, _layer, _row, _column, _tensorElement % inputLen)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }

      x.fieldShape.dimensions match {
        case 0 => _writeTensorElement(_out0, bElem, _tensorElement)
        case 1 => _writeTensorElement(_out0, bElem, _column, _tensorElement)
        case 2 => _writeTensorElement(_out0, bElem, _row, _column, _tensorElement)
        case 3 => _writeTensorElement(_out0, bElem, _layer, _row, _column, _tensorElement)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }
    }
    out
  }

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (x, batchSize) = x1
    val (b, biasBatchSize) = x2
    require(biasBatchSize == 1)
    require(grad.fieldType == x.fieldType)
    require(x.tensorShape(0) % batchSize == 0)
    val inputLen = x.tensorShape(0) / batchSize
    val biasLen = b.tensorShape(0)
    require(inputLen == biasLen)

    val sumAcrossBatchesType = new FieldType(x.fieldShape, Shape(biasLen), Float32)
    val sumAcrossBatches = GPUOperator(sumAcrossBatchesType, "sumAcrossBatches") {
      _globalThreads(x.fieldShape, Shape(biasLen))

      val curBatch = _intVar()
      val accum = _floatVar()
      accum := 0f
      _for(curBatch := 0, curBatch < batchSize, curBatch += 1) {
        val curElem = curBatch * biasLen + _tensorElement
        val elemVal = grad.fieldShape.dimensions match {
          case 0 => _readTensorElement(grad, curElem)
          case 1 => _readTensorElement(grad, _column, curElem)
          case 2 => _readTensorElement(grad, _row, _column, curElem)
          case 3 => _readTensorElement(grad, _layer, _row, _column, curElem)
          case _ => throw new RuntimeException("Invalid dimensionality")
        }
        accum += elemVal
      }

      sumAcrossBatchesType.fieldShape.dimensions match {
        case 0 => _writeTensorElement(_out0, accum, _tensorElement)
        case 1 => _writeTensorElement(_out0, accum, _column, _tensorElement)
        case 2 => _writeTensorElement(_out0, accum, _row, _column, _tensorElement)
        case 3 => _writeTensorElement(_out0, accum, _layer, _row, _column, _tensorElement)
        case _ => throw new RuntimeException("Invalid dimensionality")
      }
    }
    if (sharedBias) fieldReduceSum(sumAcrossBatches)
    else sumAcrossBatches
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, weights, sharedBias)
}

/** Factory object- eliminates clutter of 'new' operator. */
object Bias {
  /** A bias transformation that adds bias weights to the input signal
    *
    * @param input      The input signal
    * @param weights    The bias weights
    * @param sharedBias Flag for sharing bias across the input field. Sharing bias causes the
    *                   bias to be applied uniformly across the field points of the input and
    *                   requires that `weights` has a 0D field shape. Unshared bias applies a different
    *                   bias to each field point and requires `weights` to have the same field shape as `input`.
    */
  def apply (input: DifferentiableField, weights: DifferentiableField, sharedBias: Boolean = false) =
    new Bias(input, weights, sharedBias)
}
