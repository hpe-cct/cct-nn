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

/** A test of the ByteFileSensor.
  *
  * @author Dick Carter
  */
import java.io.{File, FileOutputStream}

import cogx.utilities.Random
import libcog._
import org.junit.runner.RunWith
import org.scalatest.{FunSuite, Matchers}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ByteFilePackedSensorSpec extends FunSuite with Matchers {

  /** generate data for the test. */
  def makeRandomData(length: Int): Array[Byte] = {
    val rnd = new Random
    val array = new Array[Byte](length)
    rnd.nextBytes(array)
    array
  }

  /** Create an automatically deleted file with the test data. */
  def makeTempInputFile(baseName: String, suffix: String, data: Array[Byte]): File = {
    val file = File.createTempFile(baseName, suffix)
    file.deleteOnExit()
    val outStream = new FileOutputStream(file)
    try {
      outStream.write(data)
    }
    finally
      outStream.close()
    file
  }

  /** Create a name for the temporary file.  Java appends a unique id, but start with some meaningful identifiers. */
  def mkFileName(fieldShape: Shape, tensorElements: Int, batchSize: Int, numBatches: Int,
                 extraImages: Int, pipelined: Boolean): String = {
    val testName = this.getClass().getSimpleName
    val shape = fieldShape.toArray.mkString("_")
    s"$testName-$shape-$tensorElements-$batchSize-$numBatches-$extraImages-$pipelined-id"
  }

  /** Create a vector out of the test data for the field position of the sensor ("elementIndex") as expected for
    * the given step.
    */
  def expectedVector(data:Array[Byte], fieldShape: Shape, tensorElements: Int, batchSize: Int, numBatches: Int,
                     stepNum: Int, elementIndex: Int): Vector = {
    val batchNum = stepNum % numBatches
    val elementsPerBatch = fieldShape.points * tensorElements * batchSize
    val elementsPerImage = fieldShape.points * tensorElements
    def index(imageIndex: Int, elementIndex: Int, vectorIndex: Int) =
      batchNum * elementsPerBatch + imageIndex * elementsPerImage + elementIndex * tensorElements + vectorIndex
    Vector(tensorElements * batchSize, i => {
      val imageIndex = i / tensorElements
      val vectorIndex = i % tensorElements
      data(index(imageIndex, elementIndex, vectorIndex)) & 0xFF
    })
  }

  /** The test routine, parameterized by field and test set parameters. */
  def performTest(fieldShape: Shape, tensorElements: Int, batchSize: Int, numBatches: Int,
                  extraImages: Int, pipelined: Boolean): Unit = {
    require(extraImages < batchSize, s"ExtraImages $extraImages must be less than batchSize $batchSize.")
    val numImages = numBatches * batchSize + extraImages
    val imageElements = fieldShape.points * tensorElements
    val sensorElements =  imageElements * batchSize
    val fileElements = imageElements * numImages
    // data for the test
    val data = makeRandomData(fileElements)
    // temporary input file containing the data
    val fileName = mkFileName(fieldShape, tensorElements, batchSize, numBatches, extraImages, pipelined)
    val file = makeTempInputFile(fileName, ".tmp", data)
    val path = file.getAbsolutePath

    val cg = new ComputeGraph {
      val sensor = new ByteFilePackedSensor(path, "", fieldShape, tensorElements,
        None, batchSize, updatePeriod = 1, headerLen = 0, pipelined).sensor
      probe(sensor)
    }
    import cg._

    withRelease {
      reset
      val v = new Vector(tensorElements * batchSize)

      val numSteps = numBatches + 1     // Test wrap-around to the next epoch

      for (stepNum <- 0 until numSteps) {
        val sensorReader = read(sensor).asInstanceOf[VectorFieldReader]
        sensor.dimensions match {
          case 0 =>
            sensorReader.read(v)
            require(v == expectedVector(data, fieldShape, tensorElements, batchSize, numBatches, stepNum, 0))
          case 1 =>
            val columns = fieldShape(0)
            for (column <- 0 until fieldShape(0)) {
              sensorReader.read(column, v)
              require(v == expectedVector(data, fieldShape, tensorElements, batchSize, numBatches, stepNum, column))
            }
          case 2 =>
            val (rows, columns) = (fieldShape(0), fieldShape(1))
            for (row <- 0 until rows) {
              for (column <- 0 until columns) {
                val elementIndex = row * columns + column
                sensorReader.read(row, column, v)
                require(v == expectedVector(data, fieldShape, tensorElements, batchSize, numBatches, stepNum, elementIndex))
              }
            }
          case 3 =>
            val (layers, rows, columns) = (fieldShape(0), fieldShape(1), fieldShape(2))
            for (layer <- 0 until layers) {
              for (row <- 0 until rows) {
                for (column <- 0 until columns) {
                  val elementIndex = layer * rows * columns + row * columns + column
                  sensorReader.read(layer, row, column, v)
                  require(v == expectedVector(data, fieldShape, tensorElements, batchSize, numBatches, stepNum, elementIndex))
                }
              }
            }
          case x => throw new RuntimeException(s"Improper dimensionality $x")
        }
        step
      }
    }
  }

  val pipelinedTests = Seq(false, true)

  test("0D ByteFilePackedSensor") {
    for (pipelined <- pipelinedTests) {
      performTest(Shape(), 8, 8, 1, 0, pipelined)
      performTest(Shape(), 7, 9, 2, 1, pipelined)
    }
  }

  test("1D ByteFilePackedSensor") {
    for (pipelined <- pipelinedTests) {
      performTest(Shape(16), 8, 8, 1, 0, pipelined)
      performTest(Shape(1), 1, 1, 1, 0, pipelined)
      performTest(Shape(1024), 4, 128, 2, 3, pipelined)
    }
  }

  test("2D ByteFilePackedSensor") {
    for (pipelined <- pipelinedTests) {
      performTest(Shape(256,256), 3, 16, 2, 7, pipelined)
      performTest(Shape(1,1), 3, 16, 2, 7, pipelined)
    }
  }

  test("3D ByteFilePackedSensor") {
    for (pipelined <- pipelinedTests) {
      performTest(Shape(3,16,16), 8, 8, 1, 0, pipelined)
      performTest(Shape(1,1,1), 8, 8, 1, 0, pipelined)
    }
  }

  test("ByteFilePackedSensors with corner-case shapes") {
    // Sizes not divisible by 4 (important for when bytes are packed into Floats)
    for (pipelined <- pipelinedTests) {
      performTest(Shape(15,15), 3, 5, 7, 2, pipelined)
      performTest(Shape(9,9), 1, 5, 7, 2, pipelined)

      // Try 1 for batchSize, numBatches and some field dimensions
      performTest(Shape(1,14), 1, 1, 7, 0, pipelined)
      performTest(Shape(6,1), 5, 1, 7, 0, pipelined)
    }
  }
}