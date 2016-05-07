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

package toolkit.neuralnetwork.util

import libcog._
import toolkit.neuralnetwork.operator.sumSpray


object Classify {
  /** Calculate the classification from an inference by taking the softmax of the inference
    * and comparing the max output value to the second largest. If the max value is greater
    * than the second by greater than the defined margin, then a one-hot code is emitted.
    * Otherwise an the output for that example is all zeros, indicating a failure to
    * classify above the defined margin.
    *
    * @param inference field to perform classification with
    * @param batchSize number of examples in the batched field
    * @param margin margin to use
    * @return A one-hot classification code for each example if the margin is met
    */
  def apply(inference: Field, batchSize: Int, margin: Float) = {
    val inferenceLen = inference.tensorShape(0)
    require(inference.tensorShape.dimensions == 1)
    require(inferenceLen % batchSize == 0)
    val numClasses = inferenceLen / batchSize

    val softmax = exp(inference) / max(sumSpray(exp(inference), batchSize), 1e-7f)
    val maxVal = blockReduceMax(softmax, numClasses)
    val comparison = GPUOperator(softmax.fieldType) {
      _globalThreads(softmax.fieldShape, softmax.tensorShape)
      val curBatch = _tensorElement / numClasses
      val curVal = _readTensorElement(softmax, _tensorElement)
      val curMax = _readTensorElement(maxVal, curBatch)
      _writeTensorElement(_out0, curVal > (curMax - margin), _tensorElement)
    }
    val winnerCount = blockReduceSum(comparison, numClasses)
    val classification = GPUOperator(softmax.fieldType) {
      _globalThreads(softmax.fieldShape, softmax.tensorShape)
      val curBatch = _tensorElement / numClasses
      val curVal = _readTensorElement(comparison, _tensorElement)
      val singleWinner = _readTensorElement(winnerCount, curBatch) === 1f
      _writeTensorElement(_out0, curVal * singleWinner, _tensorElement)
    }
    classification
  }
}
