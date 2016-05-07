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
private [neuralnetwork] object cropFilters {
  def apply(filters:VectorField, newShape:Shape) = {
    val filterRows = newShape(0)
    val filterColumns = newShape(1)
    val bigRows = filters.fieldShape(0)
    val bigColumns = filters.fieldShape(1)
    val outputType = new FieldType(newShape, filters.tensorShape, Float32)
    GPUOperator(outputType, "cropFilters"){
      _globalThreads(newShape, filters.tensorShape)
      val inputRow = (bigRows + _row - (filterRows-1)/2) % bigRows
      val inputColumn = (bigColumns + _column - (filterColumns-1)/2) % bigColumns

      val out = _readTensorElement(filters, inputRow, inputColumn, _tensorElement)
      _writeTensorElement(_out0, out, _row, _column, _tensorElement)
    }
  }
}
