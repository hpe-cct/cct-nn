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

/** Sum across all tensor elements in a given batch vector, then spray the result back out to the full input size
  * This operator is self-adjoint
  * @author Matthew Pickett
  */
private [neuralnetwork] object sumSpray {
  def apply (x:VectorField, batchSize:Int) = {
    require(x.tensorShape(0) % batchSize == 0)
    val inputLen = x.tensorShape(0)/batchSize
    val sum = blockReduceSum(x, inputLen)
    GPUOperator(x.fieldType, "SumSpray"){
      _globalThreads(x.fieldShape, x.tensorShape)
      val curBatch = _tensorElement / inputLen
      val curSum = _readTensorElement(sum, curBatch)
      _writeTensorElement(_out0, curSum, _tensorElement)
    }
  }
}
