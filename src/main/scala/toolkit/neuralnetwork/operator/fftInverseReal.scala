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

package toolkit.neuralnetwork.operator

import libcog._

/**
 * @author Matthew Pickett
 */
private [neuralnetwork] object fftInverseReal extends Logarithm{
  def apply(x:(Field, Field)) = {
    val xR = x._1
    val xI = x._2
    require(xR.fieldType == xI.fieldType)
    require(xR.tensorOrder == 1)
    val rows = xR.fieldShape(0)
    val columns = xR.fieldShape(1)
    require(isPowerOf2(columns))
    require(isPowerOf2(rows-1))

    val xPackedF = halfSized2packedRI(x)
    val xPacked = fftInverseRI(xPackedF._1, xPackedF._2)
    unpackRI(xPacked, xR.tensorShape(0))
  }

  private def unpackRI(x:(Field,Field), outputLen:Int) = {
    val x1 = x._1
    val x2 = x._2
    require(x1.fieldType == x2.fieldType)
    require(x1.tensorOrder==1)
    val inShape = x1.fieldShape
    val inLen = x1.tensorShape(0)
    require((outputLen+1)/2 == inLen)

    val outputType = new FieldType(inShape, Shape(outputLen), Float32)
    GPUOperator(outputType, "unpackRI"){
      _globalThreads(inShape, Shape(inLen))
      val x1cur = _readTensorElement(x1, _row, _column, _tensorElement)
      val x2cur = _readTensorElement(x2, _row, _column, _tensorElement)
      _writeTensorElement(_out0, x1cur, _row, _column, 2*_tensorElement)
      _if(2*_tensorElement+1 < outputLen){
        _writeTensorElement(_out0, x2cur, _row, _column, 2*_tensorElement+1)
      }
    }
  }

  private def halfSized2packedRI(x:(Field, Field)) = {
    val xR = x._1
    val xI = x._2
    require(xR.fieldType == xI.fieldType)
    require(xR.tensorShape.dimensions == 1)
    val inRows = xR.fieldShape(0)
    val inColumns = xR.fieldShape(1)
    val inLen = xR.tensorShape(0)
    val outRows = 2*(inRows - 1)
    val outColumns = inColumns
    val outShape = Shape((inLen+1)/2)
    val outType = new FieldType(Shape(outRows, outColumns), outShape, Float32)
    GPUOperator(outType, outType, "halfSized2packedRI") {
      _globalThreads(Shape(inRows, inColumns), outShape)
      val R1 = _readTensorElement(xR, _row, _column, 2*_tensorElement)
      val I1 = _readTensorElement(xI, _row, _column, 2*_tensorElement)
      val R2 = _floatVar()
      val I2 = _floatVar()
      _if(2*_tensorElement+1 < inLen){
        R2 := _readTensorElement(xR, _row, _column, 2*_tensorElement+1)
        I2 := _readTensorElement(xI, _row, _column, 2*_tensorElement+1)
      }
      _else{
        R2 := 0f
        I2 := 0f
      }
      _writeTensorElement(_out0, R1-I2, _row, _column, _tensorElement)
      _writeTensorElement(_out1, I1+R2, _row, _column, _tensorElement)
      _writeTensorElement(_out0, R1+I2, (outRows - _row) % outRows, (outColumns - _column) % outColumns, _tensorElement)
      _writeTensorElement(_out1, R2-I1, (outRows - _row) % outRows, (outColumns - _column) % outColumns, _tensorElement)
    }
  }
}
