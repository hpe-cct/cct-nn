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

class ByteDataSource private[ByteDataSource] (path: String,
                          fieldShape: Shape,
                          vectorLen: Int,
                          override val batchSize: Int,
                          fieldCount: Option[Long],
                          updatePeriod: Int,
                          headerLen: Int,
                          resourcePath: String,
                          pipelined: Boolean,
                          offset: Int,
                          stride: Int,
                          resetState: Long) extends DifferentiableField {

  private val sensor = new ByteFilePackedSensor(
    path,
    resourcePath,
    fieldShape,
    vectorLen,
    fieldCount,
    batchSize,
    updatePeriod,
    headerLen,
    pipelined,
    offset,
    stride,
    resetState
  ).sensor

  override val forward: Field = sensor / 255f
  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (path, fieldShape, vectorLen, batchSize, fieldCount, updatePeriod, headerLen, resourcePath,
      pipelined, offset, stride, resetState)
}

object ByteDataSource {

  def apply(path: String,
            fieldShape: Shape,
            vectorLen: Int,
            batchSize: Int,
            fieldCount: Option[Long] = None,
            updatePeriod: Int = 1,
            headerLen: Int = 0,
            resourcePath: String = "src/main/resources/",
            pipelined: Boolean = true,
            offset: Int = 0,
            stride: Int = 1,
            resetState: Long = 0L) =
    new ByteDataSource(path, fieldShape, vectorLen, batchSize, fieldCount, updatePeriod, headerLen, resourcePath,
      pipelined, offset, stride, resetState)
}
