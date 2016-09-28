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

/** A source that reads a single 32-bit float from a file and converts it to
  * a one-hot class ID code as a 0D `VectorField`. Multiple batches yield
  * a single 0D 'VectorField' with the one-hot codes stacked in the Vector
  * domain. An updatePeriod greater than one will keep the code constant
  * for multiple Cog ticks and is generally used in cases where inference
  * takes place over multiple ticks due to sampling. When the Sensor hits the
  * end of the file it will reset to the beginning automatically.
  *
  * As an example let's say that a binary file consists of the following
  * bytes: `{0x00, 0x01, 0x03, 0x00}`
  *
  * Using parameters `numClasses = 5, batchSize = 1, updatePeriod = 1`, on cog tick t
  * the source Vector would equal:
  * {{{
  * t=0, {1,0,0,0,0}
  * t=1, {0,1,0,0,0}
  * t=2, {0,0,0,1,0}
  * t=3, {1,0,0,0,0}
  * }}}
  *
  * For parameters `numClasses = 4, batchSize = 2, updatePeriod = 1`, the source
  * field would equal
  * {{{
  * t=0, {1,0,0,0,0,1,0,0}
  * t=1, {0,0,0,1,1,0,0,0}
  * }}}
  *
  * For parameters `numClasses = 4, batchSize = 2, updatePeriod = 2`, the source
  * field would equal
  * {{{
  * t=0, {1,0,0,0,0,1,0,0}
  * t=1, {1,0,0,0,0,1,0,0}
  * t=2, {0,0,0,1,1,0,0,0}
  * t=3, {0,0,0,1,1,0,0,0}
  * }}}
  *
  * Note that the length of the file is determined at Cog initialization and since it only
  * makes sense to work with an integer number of full batches some examples may be dropped
  * depending on the batchSize that you choose. For example, say that you have 107 examples
  * and a batchSize of 10. The last 7 examples will be dropped from the rotation and on the
  * 11th Cog tick, the Sensor will display the first 10 examples. If this is the case a warning
  * is printed to the console.
  *
  * The loader will search for the desired resource in three places. First, a file
  * on the filesystem at `path`. Second, a file on the filesystem at `resourcePath + path`.
  * Third, a resource in the classpath at `path`. The value of `resourcePath` will typically
  * be "src/main/resources/". Note that `fieldCount` is mandatory when loading data from the
  * classpath. When loading resources directly from the filesystem, the loader will ignore
  * `fieldCount` and use the actual length of the file.
  *
  * @param path The resource to load
  * @param numClasses Total number of classes
  * @param batchSize Number of examples to read per field
  * @param fieldCount An optional number of fields in the resource
  * @param updatePeriod Number of ticks to wait before updating the Field
  * @param headerLen Number of bytes to drop from the beginning of the file
  * @param resourcePath The search prefix for resources
  * @param bigEndian True if the file uses big endian byte order, false for little endian
  * @param offset Starting offset (in terms of batches) into the file
  * @param stride Distance (in batchSize units) between the start of one
  *               batch and the start of the next
  */
class FloatLabelSource private[FloatLabelSource] (path: String,
                                                  numClasses: Int,
                                                  override val batchSize: Int,
                                                  fieldCount: Option[Long],
                                                  updatePeriod: Int,
                                                  headerLen: Int,
                                                  resourcePath: String,
                                                  bigEndian: Boolean,
                                                  pipelined: Boolean,
                                                  offset: Int,
                                                  stride: Int,
                                                  resetState: Long) extends DifferentiableField {
  require(updatePeriod >= 1,
    "updatePeriod must be positive"
  )

  val sensor = new FloatFileSensor(
    path,
    resourcePath,
    Shape(),
    1,
    fieldCount,
    batchSize,
    updatePeriod,
    headerLen,
    bigEndian,
    pipelined,
    offset,
    stride,
    resetState).sensor

  override val forward: Field = indexToOneHotCode(sensor, numClasses)

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (path, numClasses, batchSize, fieldCount, updatePeriod, headerLen, resourcePath, bigEndian,
      pipelined, offset, stride, resetState)
}

object FloatLabelSource {
  def apply(path: String,
            numClasses: Int,
            batchSize: Int,
            fieldCount: Option[Long] = None,
            updatePeriod: Int = 1,
            headerLen: Int = 0,
            resourcePath: String = "src/main/resources/",
            bigEndian: Boolean = true,
            pipelined: Boolean = true,
            offset: Int = 0,
            stride: Int = 1,
            resetState: Long = 0L) =
    new FloatLabelSource(path, numClasses, batchSize, fieldCount, updatePeriod, headerLen, resourcePath, bigEndian,
      pipelined, offset, stride, resetState)

}
