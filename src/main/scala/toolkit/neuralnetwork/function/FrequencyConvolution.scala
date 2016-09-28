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
import toolkit.neuralnetwork.operator.{fourierBackProject, fourierFilterGrad, fourierProject}


object FrequencyConvolution extends Logarithm {

  private class UnsafeConvolution(input: DifferentiableField, weights: DifferentiableField) extends DifferentiableField {
    private val (batchSetSize, filterSetSize, inputSetSize, filterGradFilterSetSize) =
      if (Convolution.tuneForNvidiaMaxwell)
        (12, 4, 6, 8)
      else
        (4, 4, 4, 4)

    override val batchSize: Int = input.batchSize
    override val forward: libcog.Field = fourierProject(input.forward, weights.forward, batchSize, batchSetSize, filterSetSize)
    override val inputs: Map[Symbol, GradientPort] = Map(
      'input -> GradientPort(input, jacobian1, jacobianAdjoint1),
      'weights -> GradientPort(weights, jacobian2, jacobianAdjoint2))

    private def jacobian1(dx1: Field): Field =
      fourierProject(dx1, weights.forward, batchSize, batchSetSize, filterSetSize)

    private def jacobianAdjoint1(grad: Field): Field =
      fourierBackProject(grad, weights.forward, batchSize, batchSetSize, filterSetSize)

    private def jacobian2(dx2: Field): Field =
      fourierProject(input.forward, dx2, batchSize, batchSetSize, filterSetSize)

    private def jacobianAdjoint2(grad: Field): Field =
      fourierFilterGrad(input.forward, grad, weights.forward.fieldShape, batchSize, inputSetSize, filterGradFilterSetSize)

    // If you add/remove constructor parameters, you should alter the toString() implementation. */
    /** A string description of the instance in the "case class" style. */
    override def toString = this.getClass.getName +
      (input, weights)
  }

  def apply(input: DifferentiableField, weights: DifferentiableField,
            border: BorderPolicy, stride: Int = 1): DifferentiableField = {

    val filterShape = weights.forward.fieldShape
    require(input.forward.tensorShape.dimensions == 1, s"input must be a VectorField, got ${input.forward.fieldType}")
    require(input.forward.tensorShape(0) % input.batchSize == 0,
      s"number of input planes (${input.forward.tensorShape(0)}) must be an integer multiple of the batch size (${input.batchSize})")
    val inputLen = input.forward.tensorShape(0) / input.batchSize

    def errorString(s: String) = s"convolutional filter bank must contain $s filters: $filterShape"

    require(filterShape.dimensions == 2, errorString("2D"))
    require(filterShape(0) == filterShape(1), errorString("square"))
    require(filterShape(0) % 2 == 1, errorString("odd-sized"))
    val rowHalo = (filterShape(0) - 1) / 2
    val columnHalo = (filterShape(1) - 1) / 2

    def fftPad(edgeSize: Int, minPad: Int) = {
      val fftEdgeSize = roundUpPowerOf2(edgeSize + minPad)
      val padding = fftEdgeSize - edgeSize
      padding
    }

    border match {
      case BorderZero =>
        require(input.forward.fieldType.rows % stride == 0 && input.forward.fieldType.columns % stride == 0,
          s"Fieldshape ${input.forward.fieldType.fieldShape} must be a multiple of stride $stride")
        val rowPad = fftPad(input.forward.fieldType.rows, rowHalo)
        val columnPad = fftPad(input.forward.fieldType.columns, columnHalo)
        val padSizes = Seq(rowPad, columnPad)
        val fftIn = RightPad(input, padSizes)
        val fftOut = new UnsafeConvolution(fftIn, weights)
        val cropped = RightCrop(fftOut, padSizes)
        val layerOut = if (stride == 1) cropped else Downsample(cropped, stride)
        layerOut
      case BorderValid =>
        val rowPad = fftPad(input.forward.fieldType.rows, 0)
        val columnPad = fftPad(input.forward.fieldType.columns, 0)
        val padSizes = Seq(rowPad, columnPad)
        val fftIn = if (padSizes == Seq(0, 0)) input else RightPad(input, padSizes)
        val fftOut = new UnsafeConvolution(fftIn, weights)
        // Valid rows go from the first valid row to one past the last valid row
        val validRows = rowHalo until (input.forward.fieldType.rows - rowHalo)
        val validColumns = columnHalo until (input.forward.fieldType.columns - columnHalo)
        require(validRows.length % stride == 0 && validColumns.length % stride == 0,
          s"Fieldshape ${input.forward.fieldType.fieldShape} after BorderValid convolution with filter $filterShape becomes " +
            s"Shape(${validRows.length},${validColumns.length}), which is not a multiple of stride $stride.")
        val cropped = Subspace(fftOut, Seq(validRows, validColumns))
        val layerOut = if (stride == 1) cropped else Downsample(cropped, stride)
        layerOut
      case BorderCyclic =>
        if (isPowerOf2(input.forward.fieldType.rows) && isPowerOf2(input.forward.fieldType.columns))
          new UnsafeConvolution(input, weights)
        else
          throw new RuntimeException(s"Frequency Convolution with BorderPolicy $border not yet supported for this input size: ${input.forward.fieldShape}")
      case BorderClamp =>
        throw new RuntimeException(s"Frequency Convolution with BorderPolicy $border not yet supported.")
      case BorderFull =>
        throw new RuntimeException(s"Frequency Convolution with BorderPolicy $border not yet supported.")
    }
  }
}
