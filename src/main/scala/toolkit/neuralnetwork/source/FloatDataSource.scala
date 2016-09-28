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

/** A `Source` that wraps a `Sensor` which reads 32-bit floats from a file or jar and formats
  * them as elements in a `VectorField`.
  * An updatePeriod greater than one will keep the code constant
  * for multiple Cog ticks and is generally used in cases where inference
  * takes place over multiple ticks due to sampling. When using a batchSize > 1, the
  * data is appended in the Vector domain. When the Sensor hits the
  * end of the file it will reset to the beginning automatically. Bytes should be ordered
  * according to the following indexing scheme:
  *
  * `exampleIndex, fieldLayer, fieldRow, fieldColumn, vectorElement`
  *
  * As an example let's say that a binary file consists of the following
  * bytes: `{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xff}`
  *
  * Using parameters `fieldShape = Shape(2, 2), vectorLen = 1, batchSize = 1`,
  * on Cog tick t the field would equal:
  * {{{
  *   t = 0, Row{ Vector{0}, Vector{3.92e-3} }
  *          Row{ Vector{7.84e-3}, Vector{1.18e-2} }
  *   t = 1, Row{ Vector{1.57e-2}, Vector{1.96e-2} }
  *          Row{ Vector{2.35e-2, Vector{1} }
  * }}}
  *
  * Using parameters `fieldShape = Shape(2), vectorLen = 2, batchSize = 1`,
  * on Cog tick t the field would equal:
  * {{{
  *   t = 0, Row{ Vector{0, 3.92e-3}, Vector{7.84e-3, 1.18e-2} }
  *   t = 1, Row{ Vector{1.57e-2, 1.96e-2}, Vector{2.35e-2, 1} }
  * }}}
  *
  * Using parameters `fieldShape = Shape(2), vectorLen = 2, batchSize = 2`,
  * on Cog tick t the field would equal:
  * {{{
  *   t = 0, Row{ Vector{0, 3.92e-3, 1.57e-2, 1.96e-2}, Vector{7.84e-3, 1.18e-2, 2.35e-2, 1} }
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
  * The offset and stride parameters are typically used when multiple graphs
  * are cooperating to train on a single data set. In that case, these
  * parameters can be set such that each graph trains on a different part of
  * the input data set and doesn't duplicate the effort of other graphs. E.g.,
  * a stride of two skips every other batch in the input file, so a sensor
  * with offset zero and stride two hits all the even-numbered batches in the
  * file, and a second sensor with offset one and stride two hits all the odd-
  * numbered batches.
  *
  * @param path The resource to load
  * @param fieldShape The `Shape` of the field
  * @param vectorLen Number of elements in the `Vector`
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
class FloatDataSource private[FloatDataSource] (path: String,
                                                fieldShape: Shape,
                                                vectorLen: Int,
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
  private val sensor = new FloatFileSensor(
    path,
    resourcePath,
    fieldShape,
    vectorLen,
    fieldCount,
    batchSize,
    updatePeriod,
    headerLen,
    bigEndian,
    pipelined,
    offset,
    stride,
    resetState
  ).sensor

  override val forward: Field = sensor
  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (path, fieldShape, vectorLen, batchSize, fieldCount, updatePeriod, headerLen, resourcePath,
      bigEndian, pipelined, offset, stride, resetState)
}

object FloatDataSource {
  def apply(path: String,
            fieldShape: Shape,
            vectorLen: Int,
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
    new FloatDataSource(path, fieldShape, vectorLen, batchSize, fieldCount, updatePeriod, headerLen, resourcePath,
      bigEndian, pipelined, offset, stride, resetState)
}


