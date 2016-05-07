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
private [neuralnetwork] object indexToOneHotCode {
  def apply(x:Field, classes:Int) = {
    require(x.tensorShape.dimensions == 1)
    val batchSize = x.tensorShape(0)
    val outType = new FieldType(x.fieldShape, Shape(classes*batchSize), Float32)
    GPUOperator(outType, "indexToOneHotCode") {
      _globalThreads(Shape(), Shape(classes*batchSize))
      val curBatch = _tensorElement / classes
      val curClass = _tensorElement % classes
      x.fieldShape.dimensions match{
        case 0 =>
          val labelClass = _readTensorElement(x, curBatch)
          _if(curClass === labelClass) {
            _writeTensorElement(_out0, 1f, _tensorElement)
          }
          _else {
            _writeTensorElement(_out0, 0f, _tensorElement)
          }
        case 1 =>
          val labelClass = _readTensorElement(x, _column, curBatch)
          _if(curClass === labelClass) {
            _writeTensorElement(_out0, 1f, _column, _tensorElement)
          }
          _else {
            _writeTensorElement(_out0, 0f, _column, _tensorElement)
          }
        case 2 =>
          val labelClass = _readTensorElement(x, _row, _column, curBatch)
          _if(curClass === labelClass) {
            _writeTensorElement(_out0, 1f, _row, _column, _tensorElement)
          }
          _else {
            _writeTensorElement(_out0, 0f, _row, _column, _tensorElement)
          }
        case 3 =>
          val labelClass = _readTensorElement(x, _layer, _row, _column, curBatch)
          _if(curClass === labelClass) {
            _writeTensorElement(_out0, 1f, _layer, _row, _column, _tensorElement)
          }
          _else {
            _writeTensorElement(_out0, 0f, _layer, _row, _column, _tensorElement)
          }
        case _ => throw new RuntimeException("Unknown dimensionality")
      }
    }
  }
}
