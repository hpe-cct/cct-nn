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

/** The max pooling function with a `poolSize` x `poolSize` input field stepped by `stride`. The input
  * must be two dimensional.
  *
  * The ability to accomodate overlapping pools (where poolSize != stride) adds considerable complexity.  In
  * particular, the jacobianAdjoint must be prepared to sum multiple gradients (dY's) because of the potential
  * spraying of values in the forward direction.  Thus, there are two different implementation strategies taken
  * for the two cases:
  *
  * Overlapped pools:
  * The jacobianAdjoint GPUOperator allocates one thread per element of the larger 'dX' field that it
  * generates to more naturally handle the summing that might occur into each such element.
  * For performance, the jacobianAdjoint GPUOperator assumes and reads in a gradient-sized "index field"
  * that contains the field offset of the input that is the maximum of the pool.
  * The forward operator leverages the existence of the "index field" to generate its output as well.
  *
  * Non-overlapping pools:
  * The jacobianAdjoint GPUOperator allocates one thread per element of the gradient 'dY' field and has
  * that thread write the dY value to the appropriate 'dX' field element (no summing required).
  * Since no "index field" is needed to help speed the jacobianAdjoint, the forward operator examines
  * its input tile and outputs the maximum in a straightforward manner.
  *
  * The current approach might run faster by using local memory, but at the risk of not being able to accommodate
  * large strides..
  *
  * @author Dick Carter
  * @param input    input signal
  * @param poolSize the edge size of the square pooling window, defaults to 2
  * @param stride   the amount by which the pooling window is stepped (in both x and y), defaults to 2
  */
class MaxPooling private[MaxPooling] (input: DifferentiableField, poolSize: Int, stride: Int) extends DifferentiableField {
  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = _forward((input.forward, batchSize))._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input,
      dx => jacobian(dx, (input.forward, batchSize)),
      grad => jacobianAdjoint(grad, (input.forward, batchSize))))

  // Returns the number of pools along an edge of the input, given the edge size.
  private def pools(edgeSize: Int) = {
    if (edgeSize <= poolSize)
      1
    else
      1 + Math.ceil((edgeSize - poolSize).toDouble / stride).toInt
  }

  // Returns whether the input edge needs padding to meet the requirements of the pools
  private def needsPadding(edgeSize: Int) = {
    edgeSize < poolSize + (pools(edgeSize) - 1) * stride
  }

  // Creates a balanced tree of _max operators.  This should have higher performance than
  // a chain of _max operators, each one using the output of the next.  It's possible that
  // the underlying OpenCL compiler would optimize such a chain of associative operations,
  // but its safer not to count on that.
  private def maxOf(elems: Array[GPUVariable]): GPUExpression = {
    if (elems.size == 1)
      elems(0)
    else {
      val midPoint = elems.size / 2
      val (leftElems, rightElems) = elems.splitAt(midPoint)
      _max(maxOf(leftElems), maxOf(rightElems))
    }
  }

  // Can a simpler approach of maxpooling be used where the pools fully populate the input with no overlap or gaps?
  private def poolsOverlap = poolSize != stride

  // Read in the input tile that determines the pooled output element at position (_row, _column)
  private def poolElements(in: Field): Array[Array[GPUVariable]] = {
    val inRows = in.rows
    val inColumns = in.columns
    val elements = Array.tabulate(poolSize, poolSize) { (rowOffset, colOffset) =>
      val rowIndex =
        if (needsPadding(inRows))
          _min(inRows - 1, _row * stride + rowOffset)
        else
          _row * stride + rowOffset
      val columnIndex =
        if (needsPadding(inColumns))
          _min(inColumns - 1, _column * stride + colOffset)
        else
          _column * stride + colOffset
      val e = _tensorElementVar(in)
      e := _readTensorElement(in, rowIndex, columnIndex, _tensorElement)
      e
    }
    elements
  }

  /** The MaxPool operator's forward function. */
  private def _forward(in: (Field, Int)): (Field, Int) = {
    if (poolsOverlap) forwardWithOverlap(in) else forwardNoOverlap(in)
  }

  /* The MaxPool operator's forward function assuming the pools overlap.  Because of the overlap, an "index field"
   * will be needed by the jacobianAdjoint.  We also create the index field here- the CogX compiler will
   * throw one of the duplicates away.
   */
  private def forwardWithOverlap(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in

    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")
    require(x.fieldShape.toArray.max >= 2, "1x1 fields are not supported")

    val inRows = x.rows
    val inColumns = x.columns

    val maxIndexField = forwardMaxIndex(x)

    val outFieldShape = Shape(pools(inRows), pools(inColumns))
    val outputType = new FieldType(outFieldShape, x.tensorShape, x.fieldType.elementType)
    val out = GPUOperator(outputType, "maxPoolWithOverlap") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val maxIndex = _intVar()
      maxIndex := _as_int(_readTensorElement(maxIndexField, _tensorElement))
      val maxRowIndex = maxIndex / inColumns
      val maxColumnIndex = maxIndex % inColumns
      val maxValue = _readTensorElement(x, maxRowIndex, maxColumnIndex, _tensorElement)
      // Write out the maximum pool element
      _writeTensorElement(_out0, maxValue, _tensorElement)
    }
    (out, batchSize)
  }

  /* The MaxPool operator's forward function assuming the pools don't overlap. */
  private def forwardNoOverlap(in: (Field, Int)): (Field, Int) = {
    val (x, batchSize) = in

    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")
    require(x.fieldShape.toArray.max >= 2, "1x1 fields are not supported")

    val inRows = x.rows
    val inColumns = x.columns

    val outFieldShape = Shape(pools(inRows), pools(inColumns))
    val outputType = new FieldType(outFieldShape, x.tensorShape, x.fieldType.elementType)
    val out = GPUOperator(outputType, "maxPoolNoOverlap") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      // Read in the elements of the pool, clamping at the border as necessary
      val elements = poolElements(x)
      // Write out the maximum pool element
      _writeTensorElement(_out0, maxOf(elements.flatten), _tensorElement)
    }
    (out, batchSize)
  }

  // Helper function for jacobianAdjoint, index of each pool's max stored as int (in a float field)
  private def forwardMaxIndex(in: Field): Field = {
    val x = in

    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")
    require(x.fieldShape.toArray.max >= 2, "1x1 fields are not supported")

    val inRows = x.rows
    val inColumns = x.columns

    val rowPools = pools(inRows)
    val columnPools = pools(inColumns)

    val outputType = new FieldType(Shape(rowPools, columnPools), x.tensorShape, x.fieldType.elementType)
    val out = GPUOperator(outputType, "maxPoolIndex") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      // Read in the elements of the pool, clamping at the border as necessary
      val elements = poolElements(x).flatten
      val maxElementOffset = _intVar()
      maxElementOffset := 0
      val maxValue = _tensorElementVar(x)
      maxValue := elements(0)
      for (i <- 1 until poolSize * poolSize) {
        _if(elements(i) > maxValue) {
          maxValue := elements(i)
          maxElementOffset := i
        }
      }
      val absoluteRow = _row * stride + maxElementOffset / poolSize
      val absoluteColumn = _column * stride + maxElementOffset % poolSize
      val maxElementAbsoluteIndex = absoluteRow * x.columns + absoluteColumn

      // Write out the maximum pool element index (i.e. from 0 to poolSize*poolSize - 1)
      _writeTensorElement(_out0, _as_float(maxElementAbsoluteIndex), _tensorElement)
    }
    out
  }

  /** The MaxPool operator's jacobian function- used to validate the jacobianAdjoint. */
  private def jacobian(dx: Field, in: (Field, Int)): Field = {
    val (x, batchSize) = in

    require(x.fieldType == dx.fieldType)
    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")
    require(x.fieldShape.toArray.max >= 2, "1x1 fields are not supported")

    val inRows = x.rows
    val inColumns = x.columns

    val outFieldShape = Shape(pools(inRows), pools(inColumns))
    val outputType = new FieldType(outFieldShape, x.tensorShape, x.fieldType.elementType)
    GPUOperator(outputType, "maxPoolJacobian") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      // Read in the elements of the pool, clamping at the border as necessary
      val elements = poolElements(x)
      val maxRowOffset = _intVar()
      maxRowOffset := 0
      val maxColumnOffset = _intVar()
      maxColumnOffset := 0
      val maxValue = _tensorElementVar(x)
      maxValue := elements(0)(0)
      for (rowOffset <- 0 until poolSize; colOffset <- 0 until poolSize) {
        _if(elements(rowOffset)(colOffset) > maxValue) {
          maxValue := elements(rowOffset)(colOffset)
          maxRowOffset := rowOffset
          maxColumnOffset := colOffset
        }
      }
      val output = _tensorElementVar(dx)
      output := _readTensorElement(dx, _row * stride + maxRowOffset, _column * stride + maxColumnOffset, _tensorElement)

      _writeTensorElement(_out0, output, _tensorElement)

    }
  }

  /*
   * Overlapping pooling windows makes the adjoint a bit tricky.  Consider an example of a 1D input of length 5
   * with a pool size of 3 and a stride of 2.  The forward processing for one set of input data would be:
   *
   *   input          output
   *
   *   +-   -+
   *   |  2  |
   *   |     |
   *   | -1  |       +-   -+
   *   |     |       |  5  |
   *   |  5  |       |     |
   *   |     |       |  5  |
   *   |  4  |       +-   -+
   *   |     |
   *   |  0  |
   *   +-   -+
   *
   *   Note that the maximum value '5' is in both input pools.  Expressed in matrix form (for this set of input data),
   *   the forward calculation looks like:
   *
   *                                                +-   -+
   *                                                |  2  |
   *                                                |     |
   *      +-   -+    +-                   -+        | -1  |
   *      |  5  |    |  0   0   1   0   0  |        |     |
   *      |     | =  |                     |   X    |  5  |
   *      |  5  |    |  0   0   1   0   0  |        |     |
   *      +-   -+    +-                   -+        |  4  |
   *                                                |     |
   *                                                |  0  |
   *                                                +-   -+
   *
   *
   * The adjoint uses a transpose of the forward-processing matrix:
   *
   *                                    +-       -+
   *                                    |  0   0  |
   *                                    |         |
   *                                    |  0   0  |     +-     -+
   *   +-                     -+        |         |     |  dY1  |
   *   |  0   0 dY1+dY2 0   0  |   =    |  1   1  |  X  |       |
   *   +-                     -+        |         |     |  dY2  |
   *                                    |  0   0  |     +-     -+
   *                                    |         |
   *                                    |  0   0  |
   *                                    +-       -+
   *
   * So, to make the point, since the input elements might be 'sprayed' to multiple output elements during the
   * forward operation, the adjoint operation might have to sum multiple deltas into the same output element
   * (the dY1+dY2 term).  For a 2D input with a poolSize of 3 and stride of 2, the elements output by the
   * jacobianAdjoint might have 1, 2 or 4 deltas summed in, depending on their position.
   *
   * In the jacobianAdjoint implementation below, the first step is to understand how many pools a given element of
   * the `in` field is part of.  Each pool is analyzed individually and the gradient corresponding to that pool added
   * the appropriate element of the `in`-sized field that is output by the jacobianAdjoint.
   *
   */

  /** The MaxPool operator's jacobian adjoint function- exists in two flavors depending on pool overlapping. */
  private def jacobianAdjoint(grad: Field, in: (Field, Int)): Field = {
    if (poolsOverlap)
      jacobianAdjointWithOverlap(grad, in)
    else
      jacobianAdjointNoOverlap(grad, in)
  }

  /* The MaxPool operator's jacobian adjoint function, used when the pools don't overlap. */
  private def jacobianAdjointNoOverlap(grad: Field, in: (Field, Int)): Field = {
    val (x, batchSize) = in

    require(poolSize == stride, "Internal error: jacobianAdjointNoOverlap called when poolSize != stride")

    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")

    val inRows = x.rows
    val inColumns = x.columns

    val expectedGradFieldShape = Shape(pools(inRows), pools(inColumns))

    require(grad.fieldShape == expectedGradFieldShape, "Internal error: unexpected backprop field shape in MaxPooling Node: " + grad.fieldShape)

    val out = GPUOperator(x.fieldType, "maxPoolJacobianAdjointNoOverlap") {
      // There is a thread per "input" element (not per gradient field element)
      _globalThreads(grad.fieldShape, grad.tensorShape)
      // Read in the elements of the pool, clamping at the border as necessary
      val elements = poolElements(x)
      val maxRowOffset = _intVar()
      maxRowOffset := 0
      val maxColumnOffset = _intVar()
      maxColumnOffset := 0
      val maxValue = _tensorElementVar(x)
      maxValue := elements(0)(0)
      for (rowOffset <- 0 until poolSize; colOffset <- 0 until poolSize) {
        _if(elements(rowOffset)(colOffset) > maxValue) {
          maxValue := elements(rowOffset)(colOffset)
          maxRowOffset := rowOffset
          maxColumnOffset := colOffset
        }
      }
      //write the current grad out to the max position field point and 0 to the other points
      val dyCur = _readTensorElement(grad, _tensorElement)
      val i = _intVar()
      val j = _intVar()
      _for(i := _row * stride, i < _min(inRows, _row * stride + poolSize), i += 1) {
        _for(j := _column * stride, j < _min(inColumns, _column * stride + poolSize), j += 1) {
          _if(i === _row * stride + maxRowOffset && j === _column * stride + maxColumnOffset) {
            _writeTensorElement(_out0, dyCur, i, j, _tensorElement)
          }
          _else {
            _writeTensorElement(_out0, 0f, i, j, _tensorElement)
          }
        }
      }
    }
    out
  }

  /* The MaxPool operator's jacobian adjoint function, used when the pools overlap. */
  private def jacobianAdjointWithOverlap(grad: Field, in: (Field, Int)): Field = {
    val (x, batchSize) = in

    require(x.fieldShape.dimensions == 2, "Currently only implemented for 2D fields")

    val inRows = x.rows
    val inColumns = x.columns

    val rowPools = pools(inRows)
    val columnPools = pools(inColumns)

    val expectedGradFieldShape = Shape(rowPools, columnPools)

    require(grad.fieldShape == expectedGradFieldShape, "Internal error: unexpected backprop field shape in MaxPooling Node: " + grad.fieldShape)

    val maxIndexField = forwardMaxIndex(x)

    val out = GPUOperator(x.fieldType, "maxPoolJacobianAdjointWithOverlap") {
      // There is a thread per "input" element (not per gradient field element)
      _globalThreads(x.fieldShape, x.tensorShape)

      // Calculate the ceiling function on a ratio of integers
      def ceilingAdivB(a: GPUExpression, b: Int) = (a + b - 1) / b

      // What gradient field elements might need to be summed into this "input" element?
      val minRowTile = _max(0, ceilingAdivB(_row - (poolSize - 1), stride))
      val maxRowTile = _min(rowPools - 1, _row / stride)

      val minColumnTile = _max(0, ceilingAdivB(_column - (poolSize - 1), stride))
      val maxColumnTile = _min(columnPools - 1, _column / stride)

      val answer = _floatVar()
      answer := 0.0f

      val rowTile = _intVar()
      val columnTile = _intVar()

      val myAbsoluteIndex = _intVar()
      myAbsoluteIndex := _row * inColumns + _column

      _for(rowTile := minRowTile, rowTile <= maxRowTile, rowTile += 1) {
        _for(columnTile := minColumnTile, columnTile <= maxColumnTile, columnTile += 1) {
          val maxIndex = _as_int(_readTensorElement(maxIndexField, rowTile, columnTile, _tensorElement))
          _if(myAbsoluteIndex === maxIndex) {
            answer += _readTensorElement(grad, rowTile, columnTile, _tensorElement)
          }
        }
      }

      _writeTensorElement(_out0, answer, _tensorElement)
    }
    out
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, poolSize, stride)
}

/** Factory object- eliminates clutter of 'new' operator. */
object MaxPooling {
  /** The max pooling function with a `poolSize` x `poolSize` input field stepped by `stride`. The input
    * must be two dimensional.
    *
    * The ability to accomodate overlapping pools (where poolSize != stride) adds considerable complexity.  In
    * particular, the jacobianAdjoint must be prepared to sum multiple gradients (dY's) because of the potential
    * spraying of values in the forward direction.  Thus, there are two different implementation strategies taken
    * for the two cases:
    *
    * Overlapped pools:
    * The jacobianAdjoint GPUOperator allocates one thread per element of the larger 'dX' field that it
    * generates to more naturally handle the summing that might occur into each such element.
    * For performance, the jacobianAdjoint GPUOperator assumes and reads in a gradient-sized "index field"
    * that contains the field offset of the input that is the maximum of the pool.
    * The forward operator leverages the existence of the "index field" to generate its output as well.
    *
    * Non-overlapping pools:
    * The jacobianAdjoint GPUOperator allocates one thread per element of the gradient 'dY' field and has
    * that thread write the dY value to the appropriate 'dX' field element (no summing required).
    * Since no "index field" is needed to help speed the jacobianAdjoint, the forward operator examines
    * its input tile and outputs the maximum in a straightforward manner.
    *
    * The current approach might run faster by using local memory, but at the risk of not being able to accommodate
    * large strides..
    *
    * @param input    input signal
    * @param poolSize the edge size of the square pooling window, defaults to 2
    * @param stride   the amount by which the pooling window is stepped (in both x and y), defaults to 2
    */
  def apply(input: DifferentiableField, poolSize: Int = 2, stride: Int = 2) =
    new MaxPooling(input, poolSize, stride)
}

