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


object CorrectCount {
  /** Count the number of correct classifications in the inference field as compared to the
    * reference field which is considered to be the correct label.
    *
    * @param inference The inference field
    * @param reference The reference field
    * @param batchSize Number of examples in both
    * @param margin The margin to use
    * @return The total number of correct classifications
    */
  def apply(inference: Field, reference: Field, batchSize: Int, margin: Float) = {
    require(inference.fieldType == reference.fieldType)

    val inferenceLen = inference.tensorShape(0)
    require(inference.tensorShape.dimensions == 1)
    require(inferenceLen % batchSize == 0)

    val classification = Classify(inference, batchSize, margin)
    reduceSum(classification * reference)
  }
}
