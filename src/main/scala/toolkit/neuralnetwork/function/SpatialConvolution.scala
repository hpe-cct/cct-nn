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

/** A convolutional transformation that applies a filter bank to a signal in the space domain.
  *
  * @author Matthew Pickett and Dick Carter
  * @param input   The signal node
  * @param weights The filter bank input, typically a TrainableState field
  * @param stride  The factor by which the output with be downsampled after the BorderValid convolution
  */
class SpatialConvolution private[SpatialConvolution] (input: DifferentiableField, weights: DifferentiableField,
                              border: BorderPolicy, stride: Int) extends DifferentiableField {

  val supportedBorders = Seq(BorderValid, BorderZero)
  require(supportedBorders.contains(border),
    s"border policy $border not supported, must be one of: ${supportedBorders.mkString(", ")}.")
  require(weights.batchSize == 1, s"weights must have a batch size of 1, got ${weights.batchSize}")

  private val x1 = (input.forward, input.batchSize)
  private val x2 = (weights.forward, weights.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: Field = _forward(x1, x2)._1
  override val inputs: Map[Symbol, GradientPort] = Map(
    'input -> GradientPort(input, dx1 => jacobian1(dx1, x1, x2), grad => jacobianAdjoint1(grad, x1, x2)),
    'weights -> GradientPort(weights, dx2 => jacobian2(dx2, x1, x2), grad => jacobianAdjoint2(grad, x1, x2)))


  private def _forward(x1: (Field, Int), x2: (Field, Int)): (Field, Int) = {
    val (in1, batchSize) = x1
    val (filter, filterBatchSize) = x2

    assert(filterBatchSize == 1, "convolutional filter bank must have a batch size of 1")
    assert(filter.fieldShape.dimensions == 2, "convolutional filter bank must contain 2D filters")
    assert(filter.fieldShape(0) == filter.fieldShape(1), "convolutional filter bank must contain square filters")
    assert(filter.fieldShape(0) % 2 == 1, "convolutional filter bank must contain odd-sized filters")

    val inputLen = in1.tensorShape(0)
    assert(inputLen % batchSize == 0,
      s"internal test error: expecting input vector depth $inputLen to be a multiple of the batchsize $batchSize")
    val numInputs = inputLen / batchSize
    val result = blockReduceSum(projectFrame(in1, filter, border, DownsampleOutputConvolution(stride), batchSize), numInputs)
    (result, batchSize)
  }

  private def jacobian1(dx1: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in1, batchSize) = x1

    _forward((dx1, batchSize), x2)._1
  }

  private def jacobianAdjoint1(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in1, batchSize) = x1
    val (filter, filterBatchSize) = x2

    val inputLen = in1.tensorShape(0)
    val gradLen = grad.tensorShape(0)
    assert(inputLen % batchSize == 0,
      s"internal test error: expecting input vector depth $inputLen to be a multiple of the batchsize $batchSize")
    assert(gradLen % batchSize == 0,
      s"internal test error: expecting gradient vector depth $gradLen to be a multiple of the batchsize $batchSize")
    val numInputs = inputLen / batchSize
    val numFilters = filter.tensorShape(0) / numInputs

    val retField = border match {
      case BorderValid =>
        blockReduceSum(backProjectFrame(grad, filter, BorderFull, UpsampleInputConvolution(stride), batchSize), numFilters)
      case BorderZero =>
        blockReduceSum(backProjectFrame(grad, filter, BorderZero, UpsampleInputConvolution(stride), batchSize), numFilters)
      case x =>
        throw new RuntimeException(s"Unexpected border policy $x.")
    }
    retField
  }

  private def jacobian2(dx2: Field, x1: (Field, Int), x2: (Field, Int)): Field =
    _forward(x1, (dx2, x2._2))._1

  private def jacobianAdjoint2(grad: Field, x1: (Field, Int), x2: (Field, Int)): Field = {
    val (in1, batchSize) = x1
    val (filter, filterBatchSize) = x2
    val (filterRows, filterColumns) = {
      require(filter.dimensions == 2, "Expecting 2D filter")
      require(filter.rows % 2 == 1 && filter.columns % 2 == 1, s"Expecting odd filter sizes, found ${filter.fieldShape}.")
      (filter.rows, filter.columns)
    }
    val inputLen = in1.tensorShape(0)
    val gradLen = grad.tensorShape(0)
    assert(inputLen % batchSize == 0,
      s"internal test error: expecting input vector depth $inputLen to be a multiple of the batchsize $batchSize")
    assert(gradLen % batchSize == 0,
      s"internal test error: expecting gradient vector depth $gradLen to be a multiple of the batchsize $batchSize")

    // Apply a 0-valued halo region around a field, as needed for BorderZero border policy.
    def addZeroHalo(f: Field, rowHalo: Int, columnHalo: Int) = {
      val expandedShape = Shape(f.rows + 2 * rowHalo, f.columns + 2 * columnHalo)
      f.expand(BorderZero, expandedShape).shiftCyclic(rowHalo, columnHalo)
    }
    val dX2 = border match {
      case BorderValid =>
        crossCorrelateFilterAdjoint(in1, grad, BorderValid, UpsampleInputConvolution(stride), batchSize)
      case BorderZero =>
        val paddedIn = addZeroHalo(in1, filterRows/2, filterColumns/2)
        crossCorrelateFilterAdjoint(paddedIn, grad, BorderValid, UpsampleInputConvolution(stride), batchSize)
      case x =>
        throw new RuntimeException(s"Unexpected border policy $x.")
    }

    val dX2Len = dX2.tensorShape(0)
    assert(dX2Len % batchSize == 0,
      s"internal test error: expecting filter adjoint vector depth $dX2Len to be a multiple of the batchsize $batchSize")

    if (batchSize == 1)
      dX2
    else
      blockReduceSum(dX2, batchSize)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, weights, border, stride)
}

/** Factory object- eliminates clutter of 'new' operator. */
object SpatialConvolution {
  /** A convolutional transformation that applies a filter bank to a signal in the space domain.
    *
    * @param input   The signal node
    * @param weights The filter bank input, typically a TrainableState field
    * @param stride  The factor by which the output with be downsampled after the BorderValid convolution
    */
  def apply(input: DifferentiableField, weights: DifferentiableField,
            border: BorderPolicy = BorderValid, stride: Int = 1) =
    new SpatialConvolution(input, weights, border, stride)
}
