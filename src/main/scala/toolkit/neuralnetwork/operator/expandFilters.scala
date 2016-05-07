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
private [neuralnetwork] object expandFilters {
  def apply(filters:VectorField, newShape:Shape) = {
    val filterRows = filters.fieldShape(0)
    val filterColumns = filters.fieldShape(1)
    val outputType = new FieldType(newShape, filters.tensorShape, Float32)
    GPUOperator(outputType, "expandFilters"){
      _globalThreads(newShape, filters.tensorShape)
      val out = _floatVar()
      val inputRow = (_row+(filterRows-1)/2) % _rows
      val inputColumn = (_column+(filterColumns-1)/2) % _columns
      _if(inputRow < filterRows && inputColumn < filterColumns){
        out := _readTensorElement(filters, inputRow, inputColumn, _tensorElement)
      }
      _else{
        out :=0f
      }
      _writeTensorElement(_out0, out, _tensorElement)
    }
  }
}
