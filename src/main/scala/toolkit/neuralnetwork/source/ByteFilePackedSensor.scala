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

import java.nio.{ByteBuffer, ByteOrder}

import libcog._

/** A Cog `Sensor` that is used to read bytes from a file or jar and format them as
  * a `VectorField`. Each byte is interpreted as an element of the `VectorField`
  * and thus the elements of the resulting field are from 0f to 255f.
  * An updatePeriod greater than one will keep the code constant
  * for multiple Cog ticks and is generally used in cases where inference
  * takes place over multiple ticks due to sampling. When the Sensor hits the
  * end of the file it will reset to the beginning automatically.
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
  * @param resourcePath The search prefix for resources
  * @param fieldShape The `Shape` of the field
  * @param vectorLen Number of elements in the `Vector`
  * @param fieldCount An optional number of fields in the resource
  * @param batchSize Number of examples to read per field
  * @param updatePeriod Number of ticks to wait before updating the Field
  * @param headerLen Number of bytes to drop from the beginning of the file
  * @param offset Starting offset (in terms of batches) into the file
  * @param stride Distance (in batchSize units) between the start of one
  *               batch and the start of the next
  */
class ByteFilePackedSensor(path: String,
                                     resourcePath: String,
                                     fieldShape: Shape,
                                     vectorLen: Int,
                                     fieldCount: Option[Long],
                                     batchSize: Int,
                                     updatePeriod: Int,
                                     headerLen: Int,
                                     pipelined: Boolean = true,
                                     offset: Int = 0,
                                     stride: Int = 1,
                                     resetState: Long = 0L) {

  private val fieldPoints = fieldShape.points.toLong * vectorLen.toLong

  def sensor = {
    val sensor = ByteFilePackedSensor(path, resourcePath, fieldPoints, fieldCount, batchSize,
      updatePeriod, headerLen, pipelined, offset, stride, resetState)
    val outputType = new FieldType(fieldShape, Shape(vectorLen * batchSize), Float32)
    val L = outputType.layers
    val R = outputType.rows
    val C = outputType.columns
    val N = vectorLen
    GPUOperator(outputType, "ReshapeByteFilePackedSource") {
      _globalThreads(fieldShape, Shape(vectorLen * batchSize))
      val readIndex =
        fieldShape.dimensions match {
          case 0 =>
            _tensorElement
          case 1 =>
            val batch = _tensorElement / vectorLen
            val element = _tensorElement % vectorLen
            val batchOffset = batch * C * N
            val columnOffset = _column * N
            val idx = batchOffset + columnOffset + element
            idx
          case 2 =>
            val batch = _tensorElement / vectorLen
            val element = _tensorElement % vectorLen
            val batchOffset = batch * R * C * N
            val rowOffset = _row * C * N
            val columnOffset = _column * N
            val idx = batchOffset + rowOffset + columnOffset + element
            idx
          case 3 =>
            val batch = _tensorElement / vectorLen
            val element = _tensorElement % vectorLen
            val batchOffset = batch * L * R * C * N
            val layerOffset = _layer * R * C * N
            val rowOffset = _row * C * N
            val columnOffset = _column * N
            val idx = batchOffset + layerOffset + rowOffset + columnOffset + element
            idx
          case _ => throw new RuntimeException("Invalid dimensionality")
        }
      val wordReadIndex = readIndex / 4
      val shiftAmount =
        if (ByteFilePackedSensor.bigEndian)
          24 - 8 * (readIndex % 4)
        else
          8 * (readIndex % 4)
      val curElement = _readTensor(sensor, wordReadIndex)
      val curElementBytes = _as_uint(curElement)
      val byteVal = (curElementBytes >> shiftAmount) & 0xFF
      val byteValAsFloat = _convert_float(byteVal)
      _writeTensorElement(_out0, byteValAsFloat, _tensorElement)
    }
  }
}

/** The 1D Sensor that sits behind the GPUOperator reshaper in a ByteFileSensor.
  * This sensor can be saved / restored.
  *
  * @author Dick Carter
  */
object ByteFilePackedSensor {

  // This flag has nothing to do with the interpretation order of bytes in the input file, nor the arrangement of
  // floats in the final reshaped field.  Basically it describes an implementation choice of how the linear bytes
  // of the file are packed into floats, and subsequently unpacked by the reshaping kernel.  In practice, there
  // seems to be little performance difference, with the edge going to big-endian, which seems to have a slightly faster
  // CPU kernel.  The flexibility to change the setting was left in mostly to illustrate how the choice effects
  // the implementation.  Bottom line, the setting doesn't really matter, so don't bother to change it.
  private val bigEndian = true

  /** The factory method for this sensor. */
  def apply(path: String,
            resourcePath: String,
            fieldPoints: Long,
            fieldCount: Option[Long],
            batchSize: Int,
            updatePeriod: Int,
            headerLen: Int,
            pipelined: Boolean,
            offset: Int = 0,
            stride: Int = 1,
            resetState: Long = 0L): Field = {

    def getStream = StreamLoader(path, resourcePath)

    //Evaluate the number of bytes in the stream
    val fileLength = StreamLoader.length(path, resourcePath,
      fieldCount.map(c => (c * fieldPoints) + headerLen.toLong)) - headerLen.toLong

    require(fileLength % fieldPoints == 0L,
      s"File length of $path ($fileLength) is not a multiple of the number of field points $fieldPoints")

    /** Number of examples in the file.  These may not all be used if not a multiple of the batchSize. */
    val numFileExamples = fileLength / fieldPoints
    /** Number of batches before repeating. */
    val numBatches = numFileExamples / batchSize / stride
    /** Number of examples to be used in the file (a multiple of the batchSize). */
    val numExamples = numBatches * batchSize * stride

    def checkForDroppedExamples(): Unit = {
      if (numExamples != numFileExamples) {
        val droppedExamples = numFileExamples - numExamples
        println(s"Warning: number of fields in file $path is not a multiple of the batch size. " +
          s"Dropping the last $droppedExamples fields.")
      }
    }

    var bufferedStream = getStream
    /** The number of states in the state sequence of this sensor. */
    val numStatesInReadLoop = numBatches * updatePeriod
    /** The index into the state sequence of this sensor; goes from 0 to numBatches * updatePeriod - 1 */
    var state = resetState % numStatesInReadLoop
    /** A counter that goes from 0 to updatePeriod-1 to count the states of the current batch's processing. */
    def periodCounter = state % updatePeriod
    /** The index of the batch being processed by this state. */
    def currentBatch = state / updatePeriod
    /** The index of the first example of the current batch.  */
    def currentPosition = currentBatch * batchSize * stride

    assert(batchSize.toLong * fieldPoints < Int.MaxValue)
    val readLen = (batchSize * fieldPoints).toInt
    /** The readArrayLen might be bigger than the readLen to accomodate conversion to a FloatBuffer. */
    val readArrayLen = 4 * ((readLen + 3)/4)
    val floatArrayLen = readArrayLen / 4
    lazy val readArray = {
      // We want to warn the user of dropped examples once, and only if the sensor is used, so do this here:
      checkForDroppedExamples()
      new Array[Byte](readArrayLen)
    }
    /** Holds floating point version of readArray. Some of these Sensors are never stepped, so use `lazy` */
    lazy val readFloatArray = new Array[Float](floatArrayLen)
    var readArrayNeedsInit = true

    lazy val readBuffer = {
      val buf = ByteBuffer.wrap(readArray)
      val endianness = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
      buf.order(endianness)
      require(buf.order() == endianness)
      buf
    }

    def resetBuffer() = rewindBuffer(resetState % numStatesInReadLoop)

    // define the reset function to rewind to the stream's reset point (not necessarily the origin)
    def rewindBuffer(rewindState: Long) {
      bufferedStream.close()
      bufferedStream = getStream
      require(bufferedStream.skip(headerLen) == headerLen, "Stream error: Could not skip header bytes")

      // Account for offset
      var offsetBytesToSkip: Long = offset * readLen
      while (offsetBytesToSkip > 0) {
        offsetBytesToSkip -= bufferedStream.skip(offsetBytesToSkip)
      }

      state = rewindState
      readArrayNeedsInit = true
      if (rewindState != 0L) {
        val bytesPerExample = fieldPoints
        val currentPositionBytes = currentPosition * bytesPerExample
        require(bufferedStream.skip(currentPositionBytes) == currentPositionBytes,
          s"Stream error: Could not skip $currentPositionBytes bytes to current position")
      }
    }

    resetBuffer()

    /** Read from the file into the 'readArray' char buffer prior to translation to floats. */
    def readIntoReadArray(): Unit = {
      val readLenActual = bufferedStream.read(readArray, 0, readLen)
      require(readLenActual == readLen, s"Actual num bytes read ($readLenActual) differs from expected ($readLen).")
      // The term "readArray(i) & 0xFF" below performs a byte -> int conversion on Java's (strangely) signed Bytes.
      readBuffer.position(0).limit(readArrayLen) // mark readBuffer as full, may include up to 3 pad bytes
      readBuffer.asFloatBuffer().get(readFloatArray)

      readArrayNeedsInit = false
    }

    /** Advance the state counter by 1, resetting the counter back to 0 and rewinding the file reader if at the end. */
    def advanceState() {
      state += 1
      if (state >= numStatesInReadLoop) {
        rewindBuffer(0L)
      } else if (state % updatePeriod == 0) {
        var bytesToSkip = readLen * (stride - 1L)
        while (bytesToSkip > 0) {
          bytesToSkip -= bufferedStream.skip(bytesToSkip)
        }
      }
    }

    /** The next data iterator for the sensor.  This may be None if we're not updating each cycle. */
    def arrayReadNext: Option[Array[Float]] = {
      if (periodCounter == 0 || readArrayNeedsInit) {
        readIntoReadArray()
        advanceState()
        Some(readFloatArray)
      }
      else {
        advanceState()
        None
      }
    }

    /** The next data iterator for the sensor.  This sources the same data if we're not updating each cycle. */
    def arrayReadNextAlways: Array[Float] = {
      if (periodCounter == 0 || readArrayNeedsInit)
        readIntoReadArray()
      advanceState()
      readFloatArray
    }

    /** The parameters that would restore this sensor to its current state. */
    def parameters = {

      val params = SaveParameters()
      params.addParam("path", path)
      params.addParam("resourcePath", resourcePath)
      params.addLongParam("fieldCount64", fieldCount match {
        case Some(x) => x
        case None => -1L
      })
      params.addIntParam("batchSize", batchSize)
      params.addIntParam("updatePeriod", updatePeriod)
      params.addIntParam("headerLen", headerLen)
      params.addIntParam("offset", offset)
      params.addIntParam("stride", stride)
      params.addBooleanParam("pipelined", pipelined)

      /** The runtime asks for 2 states upon reset if the sensor is pipelined, else 1. */
      val backupCount = if (pipelined) 2 else 1
      val backedUpState = (state - backupCount + numStatesInReadLoop) % numStatesInReadLoop
      params.addLongParam("state", backedUpState)

      params.toString()
    }

    if (pipelined) {
      new Sensor(floatArrayLen, arrayReadNext _, resetBuffer _) {
        override def restoreParameters = parameters

        // The default restoringClass object instance would identify this as an anonymous subclass of a (pipelined) Sensor.
        // We override this here to point to the SimplePipelinedTestSensor factory object (so the restore method will be found)
        override def restoringClass = ByteFilePackedSensor
      }
    }
    else {
      new UnpipelinedSensor(floatArrayLen, arrayReadNextAlways _, resetBuffer _) {
        override def restoreParameters = parameters

        // The default restoringClass object instance would identify this as an anonymous subclass of a (pipelined) Sensor.
        // We override this here to point to the SimplePipelinedTestSensor factory object (so the restore method will be found)
        override def restoringClass = ByteFilePackedSensor
      }
    }
  }

  /** The factory method used to create a sensor from its stored parameter string. */
  def restore(fieldType: FieldType, parameterString: String) = {
    require(fieldType.dimensions == 1 && fieldType.tensorShape.dimensions == 0,
      "Expecting 1D ScalarField Sensor, found " + fieldType)
    val parameters = SaveParameters(parameterString)

    val path = parameters.getParam("path")
    val resourcePath = parameters.getParam("resourcePath")
    val fieldCountLong = parameters.getLongParam("fieldCount64")
    val fieldCount = if (fieldCountLong < 0) None else Some(fieldCountLong)
    val batchSize = parameters.getIntParam("batchSize")
    val updatePeriod = parameters.getIntParam("updatePeriod")
    val headerLen = parameters.getIntParam("headerLen")
    val offset = parameters.getIntParam("offset")
    val stride = parameters.getIntParam("stride")
    val pipelined = parameters.getBooleanParam("pipelined")
    val state = parameters.getLongParam("state")
    val fieldPoints = fieldType.fieldShape.points.toLong / batchSize

    apply(path, resourcePath, fieldPoints, fieldCount, batchSize, updatePeriod, headerLen,
      pipelined, offset, stride, state)
  }
}
