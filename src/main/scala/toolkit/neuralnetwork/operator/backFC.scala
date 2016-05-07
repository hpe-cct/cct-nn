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
private [neuralnetwork] object backFC {
  def apply(y:VectorField, w:VectorField, batchSize:Int) = {
    require(y.tensorShape(0)%batchSize == 0)
    val outLen = y.tensorShape(0)/batchSize

    val inputLen = w.tensorShape(0)/outLen
    require(w.tensorShape(0)%inputLen ==0)

    val fShape = w.fieldShape
    val fType = new FieldType(fShape, Shape(inputLen*batchSize), Float32)

    // The new approach allocates one thread per output element, but this
    // will be worse than the GPUOperator approach below if there aren't enough
    // threads to keep the GPU busy.  We conservatively enable the matrix-matrix
    // multiply kernel when we know it will be a win.

    val ThreadCountThreshold = 5000

    if (inputLen*batchSize >= ThreadCountThreshold) {
      // New approach invoking Cog core matrix-matrix multiply kernel
      val inputLenAllFieldPoints = inputLen * w.fieldShape.points
      // This use of reshape was introduced in a version of this library that depends on libcog 4.3 (or later).
      // Thus, no need to warn that these reshape uses might expect the legacy (pre-4.3) behavior.
      val yReshaped = reshape(y, Shape(), Shape(batchSize, outLen), checkLegacyReshape = false)
      val wReshaped = reshape(w, Shape(), Shape(outLen, inputLenAllFieldPoints), checkLegacyReshape = false)
      val x = transform(yReshaped, wReshaped)
      val xReshaped = reshape(x, fShape, Shape(inputLen * batchSize), checkLegacyReshape = false)
      xReshaped
    }
    else {
      // Old approach that uses a better threading strategy for small outputs
      val multiplication = GPUOperator(fType, "backFC"){
        _globalThreads(fShape, Shape(inputLen*batchSize))
        val batchIndex = _tensorElement / inputLen
        val inputIndex = _tensorElement % inputLen
        val curOutput = _floatVar()
        val curWeight = _floatVar()
        val accum = _floatVar()
        accum := 0f
        val outputIndex = _intVar()
        _for(outputIndex :=0, outputIndex< outLen, outputIndex+=1){
          curOutput := _readTensorElement(y, outputIndex + outLen*batchIndex)

          curWeight := {w.fieldShape.dimensions match {
            case 0 => _readTensorElement(w, inputIndex + inputLen*outputIndex)
            case 1 => _readTensorElement(w, _column, inputIndex + inputLen*outputIndex)
            case 2 => _readTensorElement(w, _row, _column, inputIndex + inputLen*outputIndex)
            case 3 => _readTensorElement(w, _row, _column, _layer, inputIndex + inputLen*outputIndex)
          }}

          accum += curOutput*curWeight
        }
        fShape.dimensions match {
          case 0 => _writeTensorElement(_out0, accum, _tensorElement)
          case 1 => _writeTensorElement(_out0, accum, _column, _tensorElement)
          case 2 => _writeTensorElement(_out0, accum, _row, _column, _tensorElement)
          case 3 => _writeTensorElement(_out0, accum, _row, _column, _layer, _tensorElement)
        }
      }
      multiplication
    }
  }
}