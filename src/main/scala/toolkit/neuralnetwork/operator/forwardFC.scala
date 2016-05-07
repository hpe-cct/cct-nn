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
private [neuralnetwork] object forwardFC {
  def apply(x:VectorField, w:VectorField, batchSize:Int) = {
    require(x.fieldShape == w.fieldShape)
    require(x.tensorShape(0)%batchSize == 0)
    val inputLen = x.tensorShape(0)/batchSize
    val outLen = w.tensorShape(0)/inputLen
    require(w.tensorShape(0)%outLen ==0)

    // The new approach allocates one thread per output element, but this
    // will be worse than the GPUOperator approach below if there aren't enough
    // threads to keep the GPU busy.  We conservatively enable the matrix-matrix
    // multiply kernel when we know it will be a win.

    val ThreadCountThreshold = 5000

    if (outLen*batchSize >= ThreadCountThreshold) {
      // New approach invoking Cog core matrix-matrix multiply kernel
      val inputLenAllFieldPoints = inputLen * x.fieldShape.points
      // This use of reshape was introduced in a version of this library that depends on libcog 4.3 (or later).
      // Thus, no need to warn that these reshape uses might expect the legacy (pre-4.3) behavior.
      val xReshaped = reshape(x, Shape(), Shape(batchSize, inputLenAllFieldPoints), checkLegacyReshape = false)
      val wReshaped = reshape(w, Shape(), Shape(outLen, inputLenAllFieldPoints), checkLegacyReshape = false)
      val out = transform(xReshaped, transposeMatrices(wReshaped))
      val outReshaped = reshape(out, Shape(), Shape(outLen*batchSize), checkLegacyReshape = false)
      outReshaped
    }
    else {
      // Old approach that uses a better threading strategy for small outputs
      val outShape = x.fieldShape
      val outputType = new FieldType(outShape, Shape(outLen*batchSize), Float32)
      val multiplication = GPUOperator(outputType, "batchedMultiply"){
        _globalThreads(outShape, Shape(outLen*batchSize))
        val batchIndex = _tensorElement / outLen
        val outputIndex = _tensorElement % outLen
        val accum = _floatVar()
        val curInput = _floatVar()
        val curWeight = _floatVar()
        accum := 0f
        val inputIndex = _intVar()
        _for(inputIndex :=0, inputIndex< inputLen, inputIndex+=1){
          curInput := {x.fieldShape.dimensions match {
            case 0 => _readTensorElement(x, inputIndex + inputLen*batchIndex)
            case 1 => _readTensorElement(x, _column, inputIndex + inputLen*batchIndex)
            case 2 => _readTensorElement(x, _row, _column, inputIndex + inputLen*batchIndex)
            case 3 => _readTensorElement(x, _row, _column, _layer, inputIndex + inputLen*batchIndex)
          }}

          curWeight := {x.fieldShape.dimensions match {
            case 0 => _readTensorElement(w, inputIndex + inputLen*outputIndex)
            case 1 => _readTensorElement(w, _column, inputIndex + inputLen*outputIndex)
            case 2 => _readTensorElement(w, _row, _column, inputIndex + inputLen*outputIndex)
            case 3 => _readTensorElement(w, _row, _column, _layer, inputIndex + inputLen*outputIndex)
          }}

          accum += curInput*curWeight
        }
        outShape.dimensions match {
          case 0 => _writeTensorElement(_out0, accum, _tensorElement)
          case 1 => _writeTensorElement(_out0, accum, _column, _tensorElement)
          case 2 => _writeTensorElement(_out0, accum, _row, _column, _tensorElement)
          case 3 => _writeTensorElement(_out0, accum, _row, _column, _layer, _tensorElement)
        }
      }
      fieldReduceSum(multiplication)
    }

  }
}
