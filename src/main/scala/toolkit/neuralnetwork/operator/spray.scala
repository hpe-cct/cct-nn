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

/** Expand the tensor length of an input field by replicating each input tensor element `sprayFactor` times.
  *
  * @author Dick Carter
  */
private [neuralnetwork] object spray {
  def apply (x:Field, sprayFactor:Int) = {
    val inputLen = x.tensorShape.points
    val outputLen = inputLen * sprayFactor
    val outputFieldType = x.fieldType.resizeTensor(Shape(outputLen))
    GPUOperator(outputFieldType, "Spray"){
      _globalThreads(outputFieldType.fieldShape, outputFieldType.tensorShape)
      val curVal =
        if (x.tensorOrder == 0)
          _readTensor(x)
        else
          _readTensorElement(x, _tensorElement / sprayFactor)
      _writeTensorElement(_out0, curVal, _tensorElement)
    }
  }
}
