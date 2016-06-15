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

/** A Cog `Sensor` that is used to read 32-bit floats from a file or jar and format them as
  * a `VectorField`. Each float is interpreted as an element of the `VectorField`.
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
  * @param bigEndian True if the file uses big endian byte order, false for little endian
  * @param offset Starting offset (in terms of batches) into the file
  * @param stride Distance (in batchSize units) between the start of one
  *               batch and the start of the next
  */
private[source] class FloatFileSensor(path: String,
                                      resourcePath: String,
                                      fieldShape: Shape,
                                      vectorLen: Int,
                                      fieldCount: Option[Long],
                                      batchSize: Int,
                                      updatePeriod: Int,
                                      headerLen: Int,
                                      bigEndian: Boolean,
                                      pipelined: Boolean = true,
                                      offset: Int = 0,
                                      stride: Int = 1,
                                      resetState: Long = 0L) {

  private val fieldPoints = fieldShape.points.toLong * vectorLen.toLong

  def sensor = {
    val sensor = FloatFileSensor(path, resourcePath, fieldPoints, fieldCount, batchSize,
      updatePeriod, headerLen, bigEndian, pipelined, offset, stride, resetState)
    val outputType = new FieldType(fieldShape, Shape(vectorLen * batchSize), Float32)
    val L = outputType.layers
    val R = outputType.rows
    val C = outputType.columns
    val N = vectorLen
    GPUOperator(outputType, "ReshapeFloatFileSource") {
      _globalThreads(fieldShape, Shape(vectorLen * batchSize))
      fieldShape.dimensions match {
        case 0 =>
          val n = _tensorElement
          val curElement = _readTensorElement(sensor, n, 0)
          _writeTensorElement(_out0, curElement, _tensorElement)
        case 1 =>
          val batch = _tensorElement / vectorLen
          val element = _tensorElement % vectorLen
          val batchOffset = batch * C * N
          val columnOffset = _column * N
          val n = batchOffset + columnOffset + element
          val curElement = _readTensorElement(sensor, n, 0)
          _writeTensorElement(_out0, curElement, _column, _tensorElement)
        case 2 =>
          val batch = _tensorElement / vectorLen
          val element = _tensorElement % vectorLen
          val batchOffset = batch * R * C * N
          val rowOffset = _row * C * N
          val columnOffset = _column * N
          val n = batchOffset + rowOffset + columnOffset + element
          val curElement = _readTensorElement(sensor, n, 0)
          _writeTensorElement(_out0, curElement, _row, _column, _tensorElement)
        case 3 => ???
        case _ => throw new RuntimeException("Invalid dimensionality")
      }
    }

  }
}

/** The 1D Sensor that sits behind the GPUOperator reshaper in a FloatFileSensor.
  * This sensor can be saved / restored.
  *
  * @author Dick Carter
  */
object FloatFileSensor {
  /** The factory method for this sensor. */
  def apply(path: String,
            resourcePath: String,
            fieldPoints: Long,
            fieldCount: Option[Long],
            batchSize: Int,
            updatePeriod: Int,
            headerLen: Int,
            bigEndian: Boolean,
            pipelined: Boolean,
            offset: Int = 0,
            stride: Int = 1,
            resetState: Long = 0L): Field = {

    def getStream = StreamLoader(path, resourcePath)

    //Evaluate the number of bytes in the stream
    val fileLength = StreamLoader.length(path, resourcePath,
      fieldCount.map(c => (c * fieldPoints * 4L) + headerLen.toLong)) - headerLen.toLong

    require((fileLength / 4L) % fieldPoints == 0L,
      s"File length of $fileLength is not an integer multiple of the number of field points $fieldPoints")

    /** Number of examples in the file.  These may not all be used if not a multiple of the batchSize. */
    val numFileExamples = (fileLength / 4L) / fieldPoints
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

    assert(batchSize.toLong * stride * fieldPoints * 4 < Int.MaxValue)
    val readLenFloats = (batchSize * fieldPoints).toInt
    val readLen = readLenFloats.toInt * 4
    lazy val readArray = {
      // We want to warn the user of dropped examples once, and only if the sensor is used, so do this here:
      checkForDroppedExamples()
      new Array[Byte](readLen)
    }
    var readArrayNeedsInit = true
    /** Holds floating point version of readArray. Some of these Sensors are never stepped, so use `lazy` */
    lazy val readArrayAsFloats = new Array[Float](readLenFloats)

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
        val bytesPerExample = fieldPoints * 4L
        val currentPositionBytes = currentPosition * bytesPerExample
        require(bufferedStream.skip(currentPositionBytes) == currentPositionBytes,
          s"Stream error: Could not skip $currentPositionBytes bytes to current position")
      }
    }

    resetBuffer()

    /** Read from the file into the 'readArray' char buffer prior to translation to floats. */
    def readIntoReadArray(): Unit = {
      val readLenActual = bufferedStream.read(readArray)
      require(readLenActual == readLen, s"Actual num bytes read ($readLenActual) differs from expected ($readLen).")
      readArrayNeedsInit = false
    }

    /** Convert the bytes in the readArray to an Iterator[Float]. */
    def readArrayToFloatIterator() = {
      readBuffer.position(0).limit(readLen) // mark readBuffer as full
      readBuffer.asFloatBuffer().get(readArrayAsFloats)
      readArrayAsFloats.toIterator
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
    def readNext: Option[Iterator[Float]] = {
      if (periodCounter == 0 || readArrayNeedsInit) {
        readIntoReadArray()
        advanceState()
        Some(readArrayToFloatIterator())
      }
      else {
        advanceState()
        None
      }
    }

    /** The next data iterator for the sensor.  This sources the same data if we're not updating each cycle. */
    def readNextAlways: Iterator[Float] = {
      if (periodCounter == 0 || readArrayNeedsInit)
        readIntoReadArray()
      advanceState()
      readArrayToFloatIterator()
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
      params.addBooleanParam("bigEndian", bigEndian)
      params.addBooleanParam("pipelined", pipelined)

      /** The runtime asks for 2 states upon reset if the sensor is pipelined, else 1. */
      val backupCount = if (pipelined) 2 else 1
      val backedUpState = (state - backupCount + numStatesInReadLoop) % numStatesInReadLoop
      params.addLongParam("state", backedUpState)

      params.toString()
    }

    if (pipelined) {
      new Sensor(Shape(readLenFloats), readNext _, resetBuffer _) {
        override def restoreParameters = parameters

        // The default restoringClass object instance would identify this as an anonymous subclass of a (pipelined) Sensor.
        // We override this here to point to the SimplePipelinedTestSensor factory object (so the restore method will be found)
        override def restoringClass = FloatFileSensor
      }
    }
    else {
      new UnpipelinedSensor(Shape(readLenFloats), readNextAlways _, resetBuffer _) {
        override def restoreParameters = parameters

        // The default restoringClass object instance would identify this as an anonymous subclass of a (pipelined) Sensor.
        // We override this here to point to the SimplePipelinedTestSensor factory object (so the restore method will be found)
        override def restoringClass = FloatFileSensor
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
    val bigEndian = parameters.getBooleanParam("bigEndian")
    val offset = parameters.getIntParam("offset")
    val stride = parameters.getIntParam("stride")
    val pipelined = parameters.getBooleanParam("pipelined")
    val state = parameters.getLongParam("state")
    val fieldPoints = fieldType.fieldShape.points.toLong / batchSize

    apply(path, resourcePath, fieldPoints, fieldCount, batchSize, updatePeriod, headerLen,
      bigEndian, pipelined, offset, stride, state)
  }
}

