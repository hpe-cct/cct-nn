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

package toolkit.neuralnetwork.source

import libcog._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.operator.indexToOneHotCode


case class ByteLabelSource(path: String,
                           numClasses: Int,
                           override val batchSize: Int,
                           fieldCount: Option[Long] = None,
                           updatePeriod: Int = 1,
                           headerLen: Int = 0,
                           resourcePath: String = "src/main/resources/",
                           pipelined: Boolean = true,
                           offset: Int = 0,
                           stride: Int = 1,
                           resetState: Long = 0L) extends DifferentiableField {
  require(numClasses <= 256, "Number of classes must be <= 256 since each byte represents a class index")
  require(updatePeriod >= 1, "updatePeriod must be positive")

  val sensor = new ByteFilePackedSensor(
    path,
    resourcePath,
    Shape(),
    1,
    fieldCount,
    batchSize,
    updatePeriod,
    headerLen,
    pipelined,
    offset,
    stride,
    resetState).sensor

  override val forward: Field = indexToOneHotCode(sensor, numClasses)
}
