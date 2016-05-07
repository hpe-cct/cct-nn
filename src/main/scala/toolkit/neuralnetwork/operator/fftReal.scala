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
private [neuralnetwork] object fftReal {
  def apply(x:VectorField) = {
    val vectorLen = x.tensorShape(0)
    val xPacked = backFusablePackRI(x)
//    val xPacked = packRI(x)
    val xPackedF = fftRI(xPacked._1, xPacked._2)
    packedRI2halfSized(xPackedF, vectorLen)
  }

  private def packedRI2halfSized(packedF:(Field,Field), outputLen:Int) = {
    val packedFR = packedF._1
    val packedFI = packedF._2
    require(packedFR.fieldType == packedFI.fieldType)
    require(packedFR.tensorShape.dimensions==1)
    val inRows = packedFR.fieldShape(0)
    val inColumns = packedFR.fieldShape(1)
    val halfRows = inRows/2+1

    val inputLen = packedFR.tensorShape(0)
    require((outputLen+1)/2 == inputLen)

    val outType = new FieldType(Shape(halfRows, inColumns), Shape(outputLen), Float32)
    GPUOperator(outType, outType, "packedRI2halfSized") {
      //define threading shape - half plus one size field shape due to symmetry
      _globalThreads(outType.fieldShape, Shape(inputLen))

      val xFR = _readTensorElement(packedFR, _row, _column, _tensorElement)
      val xFI = _readTensorElement(packedFI, _row, _column, _tensorElement)
      val xFRFlip = _readTensorElement(packedFR, (inRows - _row) % inRows, (inColumns - _column) % inColumns, _tensorElement)
      val xFIFlip = _readTensorElement(packedFI, (inRows - _row) % inRows, (inColumns - _column) % inColumns, _tensorElement)

      //unpack the first packed batch
      val xFR0 = 0.5f*(xFR + xFRFlip)
      val xFI0 = 0.5f*(xFI - xFIFlip)

      //unpack the second packed batch
      val xFR1 = 0.5f*(xFI + xFIFlip)
      val xFI1 = 0.5f*(xFRFlip - xFR)

      _writeTensorElement(_out0, xFR0, _row, _column, _tensorElement*2)
      _writeTensorElement(_out1, xFI0, _row, _column, _tensorElement*2)
      _if(_tensorElement*2+1 < outputLen){
        _writeTensorElement(_out0, xFR1, _row, _column, _tensorElement*2+1)
        _writeTensorElement(_out1, xFI1, _row, _column, _tensorElement*2+1)
      }
    }
  }

  private def packRI(x:VectorField) = {
    val inShape = x.fieldShape
    val inLen = x.tensorShape(0)

    val packedInputType = new FieldType(inShape, Shape((x.tensorShape(0)+1)/2), Float32)
    val (reX, imX) = GPUOperator(packedInputType,packedInputType, "packRI"){
      _globalThreads(packedInputType.fieldShape, packedInputType.tensorShape)
      val a = _readTensorElement(x, _row, _column, _tensorElement*2)
      val b = _floatVar()
      _if(_tensorElement*2+1 < inLen){
        b := _readTensorElement(x, _row, _column, _tensorElement*2+1)
      }
      _else{
        b := 0f
      }

      _writeTensorElement(_out0, a, _row, _column, _tensorElement)
      _writeTensorElement(_out1, b, _row, _column, _tensorElement)
    }
    (reX,imX)
  }

  // Same function as packRI above, but uses a thread organization that allows for a
  // "local read" of the input.  This packing function generally feeds an FFT kernel,
  // which has an input read pattern that precludes merging.  Backward merging of
  // this kernel is more important to optimize for, and this implementation permits this.
  private def backFusablePackRI(x:VectorField) = {
    val inShape = x.fieldShape
    val inLen = x.tensorShape(0)
    val outLen = (x.tensorShape(0)+1)/2

    val packedInputType = new FieldType(inShape, Shape(outLen), Float32)
    val (reX, imX) = GPUOperator(packedInputType,packedInputType, "backFusablePackRI"){
      _globalThreads(x.fieldShape, x.tensorShape)
      val a = _readTensorElement(x, _tensorElement)
      _if(_tensorElement % 2 === 0) {
        _writeTensorElement(_out0, a, _row, _column, _tensorElement/2)
      }
      _else {
        _writeTensorElement(_out1, a, _row, _column, _tensorElement/2)
      }
      // Emit code to supply 0f values for "missing" input plane, but only if
      // there are an odd number of input planes.
      if (inLen % 2 == 1) {
        _if (_tensorElement === inLen - 1) {
          _writeTensorElement(_out1, 0f, _row, _column, _tensorElement/2)
        }
      }
    }
    (reX,imX)
  }
}
