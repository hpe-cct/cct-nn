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
private [neuralnetwork] object weightGradFC {
  def apply(x:VectorField, grad:VectorField, batchSize:Int) = {
    require(x.tensorShape(0)%batchSize == 0)
    val inputLen = x.tensorShape(0) / batchSize
    require(grad.tensorShape(0)%batchSize == 0)
    val gradLen  = grad.tensorShape(0) / batchSize

    val outShape = x.fieldShape
    val outType = new FieldType(outShape, Shape(inputLen*gradLen), Float32)

    // The new approach allocates one thread per output element, but this
    // will be worse than the GPUOperator approach below if there aren't enough
    // threads to keep the GPU busy.  We conservatively enable the matrix-matrix
    // multiply kernel when we know it will be a win.

    val ThreadCountThreshold = 5000

    if (inputLen*gradLen >= ThreadCountThreshold) {
      // New approach invoking Cog core matrix-matrix multiply kernel
      val inputLenAllFieldPoints = inputLen * x.fieldShape.points
      // This use of reshape was introduced in a version of this library that depends on libcog 4.3 (or later).
      // Thus, no need to warn that these reshape uses might expect the legacy (pre-4.3) behavior.
      val xReshaped = reshape(x, Shape(), Shape(batchSize, inputLenAllFieldPoints), checkLegacyReshape = false)
      val gradReshaped = reshape(grad, Shape(), Shape(batchSize, gradLen), checkLegacyReshape = false)
      val out = transform(transposeMatrices(gradReshaped), xReshaped)
      val outReshaped = reshape(out, outShape, Shape(inputLen*gradLen), checkLegacyReshape = false)
      outReshaped
    }
    else {
      // Old approach that uses a better threading strategy for small outputs
      GPUOperator(outType, "weightGradFC") {
        _globalThreads(outShape, Shape(inputLen * gradLen))
        val inputIndex = _tensorElement % inputLen
        val gradIndex = _tensorElement / inputLen
        val batchIndex = _intVar()
        val accum = _floatVar()
        accum := 0f
        val curInput = _floatVar()
        val curOutput = _floatVar()
        _for(batchIndex := 0, batchIndex < batchSize, batchIndex += 1) {
          curInput := {
            x.fieldShape.dimensions match {
              case 0 => _readTensorElement(x, inputIndex + batchIndex * inputLen)
              case 1 => _readTensorElement(x, _column, inputIndex + batchIndex * inputLen)
              case 2 => _readTensorElement(x, _row, _column, inputIndex + batchIndex * inputLen)
              case 3 => _readTensorElement(x, _row, _column, _layer, inputIndex + batchIndex * inputLen)
            }
          }

          curOutput := _readTensorElement(grad, gradIndex + batchIndex * gradLen)

          accum += curInput * curOutput
        }
        x.fieldShape.dimensions match {
          case 0 => _writeTensorElement(_out0, accum, _tensorElement)
          case 1 => _writeTensorElement(_out0, accum, _column, _tensorElement)
          case 2 => _writeTensorElement(_out0, accum, _row, _column, _tensorElement)
          case 3 => _writeTensorElement(_out0, accum, _row, _column, _layer, _tensorElement)
        }
      }
    }
  }
}
