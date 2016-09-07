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

package toolkit.neuralnetwork.examples.util

import java.io.{File, FileInputStream, FileNotFoundException}
import java.nio.{ByteBuffer, ByteOrder}

import libcog._

import scala.util.Try


/**
 * Data augmentation utilities for use by Alex Net.
 *
 * Julie Symons, 03/11/2016, Hewlett-Packard Enterprise Labs, Palo Alto, CA.
 *
 */

object DataAugmentation {

  class DoubleFileReader(path: String,
                         resourcePath: String,
                         fieldShape: Shape,
                         vectorLen: Int,
                         batchSize: Int,
                         headerLen: Int,
                         bigEndian: Boolean) {

    private val fieldPoints = fieldShape.points.toLong * vectorLen.toLong

    val stream = Try(new FileInputStream(path)) recover {
      case e: FileNotFoundException => new FileInputStream(resourcePath + path)
    } recover {
      case e: FileNotFoundException =>
        assert(this.getClass.getClassLoader.getResource(path) != null)
        this.getClass.getClassLoader.getResourceAsStream(path)
    }

    def fileLength(path: String) = {
      if (new File(path).exists())
        new File(path).length()
    }

    val fileL = fileLength(path)

    assert(batchSize.toLong * fieldPoints * 8 < Int.MaxValue)
    val readLenDoubles = (batchSize * fieldPoints).toInt
    val readLen = readLenDoubles.toInt * 8
    val readArray = new Array[Byte](readLen)
    var readArrayNeedsInit = true
    val readArrayAsDoubles = new Array[Double](readLenDoubles)

    val readBuffer = {
      val buf = ByteBuffer.wrap(readArray)
      val endianness = if (bigEndian) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
      buf.order(endianness)
      require(buf.order() == endianness)
      buf
    }

    /** Read from the file into the 'readArray' char buffer prior to translation to doubles. */
    def readIntoReadArray(): Unit = {
      val readLenActual = stream.get.read(readArray)
      require(readLenActual == readLen, s"Actual num bytes read ($readLenActual) differs from expected ($readLen).")
      readArrayNeedsInit = false
    }

    /** Convert the bytes in the readArray to an Iterator[Double]. */
    def readArrayToDoubles() = {
      readBuffer.position(0).limit(readLen) // mark readBuffer as full
      val s = readBuffer.asDoubleBuffer.get(readArrayAsDoubles)
      readArrayAsDoubles
    }
  }

  /**
   * Reads mean image as double and put it into a VectorField as Float32 and divided by 255
   *
   *
   */

  def loadOffsetVector(f: String): Array[Array[Vector]] = {

    val nRows = 256
    val nCols = 256
    val nColors = 3
    val fieldShape = Shape(nRows, nCols)
    val vectorShape = Shape(nColors)

    val meanImageFile = new File(f).toString
    val r = new DoubleFileReader(meanImageFile,
      "src/main/resources/",
      fieldShape,
      3,
      1,
      0,
      false)
    val res = r.readIntoReadArray
    val r2 = r.readArrayToDoubles

    val meanImageArray = Array.tabulate[Vector](nRows, nCols) {
      (iRow, iColumn) => Vector(nColors, iColor => r2(iRow * 768 + iColumn * 3 + iColor).toFloat / 255f)
    }

    meanImageArray
  }

  /**
   * Subtract an offset from each image in the batch
   *
   *
   */

  def subtractOffset(img: Field, offset: Field): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    require(offset.fieldShape.dimensions == 2, "Requires a 2D field")
    require(img.fieldShape == offset.fieldShape, "Requires same field shape")
    require(img.fieldType.elementType == offset.fieldType.elementType, "Requires same element type")

    val outputType = img.fieldType

    GPUOperator(outputType, "SubtractOffset") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val a = _readTensorElement(img, _tensorElement)
      val offset1 = _readTensorElement(offset, _tensorElement % 3)
      _writeTensorElement(_out0, a - offset1, _tensorElement)
    }
  }


  /**
   * Crop image
   * crop starting position provided as input parameters
   *
   */

  def cropImage(img: Field, outputShape: Shape, startY: Int, startX: Int): Field = {

    assert(img.fieldShape.dimensions == 2, "Requires a 2D field")
    assert(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    assert(img.fieldShape(1) > outputShape(1), "Requires output size < input size")
    assert(img.fieldShape(0) >= outputShape(0) + startY, "Requires starting row + output size <= input size")
    assert(img.fieldShape(1) >= outputShape(1) + startX, "Requires starting column + output size <= input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)

    GPUOperator(outputType, "cropImage") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val iY = _row
      val iX = _column
      val iYStart = _intVar()
      val iXStart = _intVar()

      iYStart := startY
      iXStart := startX

      val value = _readTensorElement(img, iY + iYStart, iX + iXStart, _tensorElement)
      _writeTensorElement(_out0, value, _tensorElement)
    }
  }


  /**
   * Crop image
   * crop starting position generated randomly by CPU operator
   * Assumes square image and square crop,
   *
   */

  def cropImage2(img: Field, outputShape: Shape): Field = {

    assert(img.fieldShape.dimensions == 2, "Requires a 2D field")
    assert(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    assert(img.fieldShape(1) > outputShape(1), "Requires output size < input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)

    val calcdRange = img.fieldShape(0) - outputShape(0)
    val range = ScalarField(2, (column) => calcdRange) // assumes square crop based on nRows, ignored for now
    val guide = RandomPatch(range) // this is currently hard-coded to 26, returns random Y and X in 0..26

    GPUOperator(outputType, "cropImage2") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val iY = _row
      val iX = _column

      val iYStart = _readTensor(guide, 0)
      val iXStart = _readTensor(guide, 1)

      val value = _readTensorElement(img, iY + iYStart, iX + iXStart, _tensorElement)
      _writeTensorElement(_out0, value, _tensorElement)
    }
  }

  Random.setDeterministic()
  val rng = new Random()
  var range = 27                                  // default is 27, but gets overridden by img.fieldShape - outputShape

  def nextValueP:Option[Iterator[Float]] = {      // random Y and X in a range
    val x = Math.abs(rng.nextInt) % range
    val y = Math.abs(rng.nextInt) % range
    val resultArray = Array[Float](x, y)
    Option(resultArray.iterator)
  }

  def nextValueB:Option[Iterator[Float]] = {      // random 0 or 1
    val x = Math.abs(rng.nextInt) % 2
    val resultArray = Array[Float](x)
    Option(resultArray.iterator)
  }

  def resetHook() = { /* Nothing to do here. */ }

  /**
   * Crop image
   * uses sensor to get random position
   * Assumes square image and square crop,
   *
   */

  def cropImage3(img: Field, outputShape: Shape): Field = {

    assert(img.fieldShape.dimensions == 2, "Requires a 2D field")
    assert(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    assert(img.fieldShape(1) > outputShape(1), "Requires output size < input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)
    range = img.fieldShape(0) - outputShape(0) + 1

    val randomCrop = new Sensor(Shape(2), nextValueP _, resetHook _)

    GPUOperator(outputType, "cropImage3") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val iY = _row
      val iX = _column

      val g1 = _intVar()
      val g2 = _intVar()

      val iYStart = _readTensor(randomCrop,0)
      val iXStart = _readTensor(randomCrop,1)

      g1:= iYStart
      g2:= iXStart

      val value = _readTensorElement(img, iY + g1, iX + g2, _tensorElement)
      _writeTensorElement(_out0, value, _tensorElement)
    }
  }



  /**
   * Take the horizontal reflect of an image
   *
   *
   */

  def horizontalReflection(img: Field): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    GPUOperator(img.fieldType, "horizontalRefection") {
      _globalThreads(img.fieldShape, img.tensorShape)
      val readRow = _row
      val readColumn = _columns - _column - 1
      val x = _readTensorElement(img, readRow, readColumn, _tensorElement)
      _writeTensorElement(_out0, x, _tensorElement)
    }
  }


  /**
   * Take the horizontal reflect of an image
   * Randomly decide whether to take the reflection or not using sensor
   *
   */

  def randomHorizontalReflection(img: Field): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")

    val randomFlip = new Sensor(Shape(1), nextValueB _, resetHook _)

    GPUOperator(img.fieldType, "randomHorizontalRefection") {
      _globalThreads(img.fieldShape, img.tensorShape)
      val readRow = _row
      val readColumn = _columns - _column - 1
      val x = _readTensorElement(img, readRow, readColumn, _tensorElement)
      _writeTensorElement(_out0, x, _tensorElement)

      val flip = _readTensor(randomFlip)

      _if(_isequal(flip, 1f)) {   // horizontal reflection
        val readRow = _row
        val readColumn = _columns - _column - 1
        val x = _readTensorElement(img, readRow, readColumn, _tensorElement)
        _writeTensorElement(_out0, x, _tensorElement)
      }
      _else {  // no horizontal reflection
        val x = _readTensorElement(img, _tensorElement)
        _writeTensorElement(_out0, x, _tensorElement)
      }
    }
  }


  /**
   * Combined GPU operator for subtractOffset, crop, and horizontal reflection
   * uses CPU operator to generate random inputs
   * if random, random crop and random h. reflection
   * if random = false, center crop, no reflection
   */

  def subtractCropReflect(img: Field, offset: Field, outputShape: Shape, random: Int = 1): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    require(offset.fieldShape.dimensions == 2, "Requires a 2D field")
    require(img.fieldShape == offset.fieldShape, "Requires same field shape")
    require(img.fieldType.elementType == offset.fieldType.elementType, "Requires same element type")
    require(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    require(img.fieldShape(1) > outputShape(1), "Requires output size < input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)

    val range = img.fieldShape(0) - outputShape(0)
    val centerCropStart : Int = range / 2
    val rangeField = ScalarField(2, (column) => range) // assumes square crop based on nRows, ignored for now
    val guide = RandomPatch(rangeField) // this is currently hard-coded to 26, returns random Y and X in 0..26

    val flipField = ScalarField(0f)
    val flipBoolean = RandomFlip(flipField)

    GPUOperator(outputType, "SubtractCropReflect") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)

      val iY = _row
      val iX = _column

      val doRandom = _floatVar()
      doRandom := random

      _if(_isequal(doRandom, 1f)) {
        // random crop and random horizontal reflection
        val iYStart = _readTensor(guide, 0)
        val iXStart = _readTensor(guide, 1)

        val flip = _readTensor(flipBoolean)

        _if(_isnotequal(flip, 0f)) {
          // don't do horizontal reflection
          val value = _readTensorElement(img, iY + iYStart, iX + iXStart, _tensorElement)
          val offset1 = _readTensorElement(offset, iY + iYStart, iX + iXStart, _tensorElement % 3)
          _writeTensorElement(_out0, value - offset1, _tensorElement)
        }
        _else {
          val readColumn = (outputShape(0) + iXStart) - iX - 1 //_columns - _column - 1
          val value = _readTensorElement(img, iY + iYStart, readColumn, _tensorElement)
          val offset1 = _readTensorElement(offset, iY + iYStart, readColumn, _tensorElement % 3)
          _writeTensorElement(_out0, value - offset1, _tensorElement)
        }
      }
      _else {
        // do center crop, don't do horizontal reflection
        val iYStart = _intVar()
        val iXStart = _intVar()
        iYStart := centerCropStart
        iXStart := centerCropStart
        val value = _readTensorElement(img, iY + iYStart, iX + iXStart, _tensorElement)
        val offset1 = _readTensorElement(offset, iY + iYStart, iX + iXStart, _tensorElement % 3)
        _writeTensorElement(_out0, value - offset1, _tensorElement)

      }
    }
  }

  /**
   * Combined GPU operator for subtractOffset, crop, and horizontal reflection
   * uses sensors to generate random inputs
   * if random, random crop and random h. reflection
   * if random = false, center crop, no reflection
   */

  def subtractCropReflect2(img: Field, offset: Field, outputShape: Shape, random: Int = 1): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    require(offset.fieldShape.dimensions == 2, "Requires a 2D field")
    require(img.fieldShape == offset.fieldShape, "Requires same field shape")
    require(img.fieldType.elementType == offset.fieldType.elementType, "Requires same element type")
    require(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    require(img.fieldShape(1) > outputShape(1), "Requires output size < input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)

    range = img.fieldShape(0) - outputShape(0) + 1
    val centerCropStart : Int = range / 2

    val randomCrop = new Sensor(Shape(2), nextValueP _, resetHook _)
    val randomFlip = new Sensor(Shape(1), nextValueB _, resetHook _)

    GPUOperator(outputType, "SubtractCropReflect2") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)

      val iY = _row
      val iX = _column

      val doRandom = _floatVar()
      doRandom := random

      _if(_isequal(doRandom, 1f)) {
        // random crop and random horizontal reflection
        val guideY = _intVar()
        val guideX = _intVar()

        val iYStart = _readTensor(randomCrop,0)
        val iXStart = _readTensor(randomCrop,1)

        guideY:= iYStart
        guideX:= iXStart

        val flip = _readTensor(randomFlip)

        _if(_isequal(flip, 1f)) {   // horizontal reflection
          val readColumn = (outputShape(0) + guideX) - iX - 1 //_columns - _column - 1
          val value = _readTensorElement(img, iY + guideY, readColumn, _tensorElement)
          val offset1 = _readTensorElement(offset, iY + guideY, readColumn, _tensorElement % 3)
          _writeTensorElement(_out0, value - offset1, _tensorElement)
        }
        _else {
          // don't do horizontal reflection
          val value = _readTensorElement(img, iY + guideY, iX + guideX, _tensorElement)
          val offset1 = _readTensorElement(offset, iY + iYStart, iX + iXStart, _tensorElement % 3)
          _writeTensorElement(_out0, value - offset1, _tensorElement)
        }
      }
      _else {
        // do center crop, don't do horizontal reflection
        val iYStart = _intVar()
        val iXStart = _intVar()
        iYStart := centerCropStart
        iXStart := centerCropStart
        val value = _readTensorElement(img, iY + iYStart, iX + iXStart, _tensorElement)
        val offset1 = _readTensorElement(offset, iY + iYStart, iX + iXStart, _tensorElement % 3)
        _writeTensorElement(_out0, value - offset1, _tensorElement)
      }
    }
  }

  /**
   * Creates patch for validation
   * GPU operator for crop, and horizontal reflection
   * For validation, we need to cycle through providing an image from a set of 10 possible images
   * There are 5 possible crop positions: 4 corners, and 1 center
   * And with or without horizontal reflection
   * A sensor is used to cycle through the 10 possible patches
   */

  val patches = Array((0f,0f,0f), (0f,26f,0f), (13f,13f,0f), (26f,0f,0f), (26f,26f,0f),
    (0f,0f,1f), (0f,26f,1f), (13f,13f,1f), (26f,0f,1f), (26f,26f,1f))

  def validationPatch(img: Field, offset: Field, outputShape: Shape): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    require(offset.fieldShape.dimensions == 2, "Requires a 2D field")
    require(img.fieldShape == offset.fieldShape, "Requires same field shape")
    require(img.fieldType.elementType == offset.fieldType.elementType, "Requires same element type")
    require(img.fieldShape(0) > outputShape(0), "Requires output size < input size")
    require(img.fieldShape(1) > outputShape(1), "Requires output size < input size")

    val outputType = new FieldType(outputShape, img.tensorShape, img.fieldType.elementType)

    class PatchIndexer {
      var count = -1

      def nextIdx: Int = {
        count += 1
        val idx = count % patches.size
        idx
      }
      def reset: Unit = {
        count = -1
      }
    }

    val idx = new PatchIndexer

    def nextIndex:Option[Iterator[Float]] = {
      //val next = Array[Float](idx.nextY, idx.nextX, idx.nextR)
      //val (x,y,z)=idx.next3
      val index = idx.nextIdx
      //println(s"nextIndex: $index")
      val x = patches (index)._1
      val y = patches (index)._2
      val z = patches (index)._3
      val next = Array[Float] (x, y, z)
      Option (next.iterator)
    }

    val nextPatch = new Sensor(Shape(3), nextIndex _, idx.reset _)

    GPUOperator(outputType, "ValidationPatch") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)

      val iY = _row
      val iX = _column

      val startY = _readTensor(nextPatch, 0)
      val startX = _readTensor(nextPatch, 1)
      val flip = _readTensor(nextPatch, 2)

      _if(_isequal(flip, 1f)) {   // horizontal reflection
        val readColumn = (outputShape(0) + startX) - iX - 1 //_columns - _column - 1
        val value = _readTensorElement(img, iY + startY, readColumn, _tensorElement)
        val offset1 = _readTensorElement(offset, iY + startY, readColumn, _tensorElement % 3)
        _writeTensorElement(_out0, value - offset1, _tensorElement)
      }
      _else {
        // don't do horizontal reflection
        val value = _readTensorElement(img, iY + startY, iX + startX, _tensorElement)
        val offset1 = _readTensorElement(offset, iY + startY, iX + startX, _tensorElement % 3)
        _writeTensorElement(_out0, value - offset1, _tensorElement)
      }
    }
  }

  def nextValueRGB:Option[Iterator[Float]] = {

    val alpha1 = 0.1*(rng.nextGaussian())
    val alpha2 = 0.1*(rng.nextGaussian())
    val alpha3 = 0.1*(rng.nextGaussian())

    val p1 = (-0.56771488, -0.58145678, -0.5827588)
    val p2 = (-0.72252667, 0.01267614, 0.69122683)
    val p3 = (0.3945314, -0.81347853, 0.42731446)

    val lambda1 = 0.204896523508
    val lambda2 = 0.0194841685264
    val lambda3 = 0.00456344555301

    val w1 = alpha1 * lambda1
    val w2 = alpha2 * lambda2
    val w3 = alpha3 * lambda3

    val rShift = (p1._1 * w1 + p2._1 * w2 + p3._1 * w3).toFloat
    val gShift = (p1._2 * w1 + p2._2 * w2 + p3._2 * w3).toFloat
    val bShift = (p1._3 * w1 + p2._3 * w2 + p3._3 * w3).toFloat

    val resultArray = Array[Float](rShift,gShift,bShift)
    Option(resultArray.iterator)
  }


  /**
   * applyColorShift GPU operator
   * gets 3 shift values from sensor
   * adds the 3 shift values for every RBG pixel in the image or batch of images
   */

  def applyColorShift(img: Field): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    val outputType = img.fieldType

    val colorShift = new Sensor(Shape(3), nextValueRGB _, resetHook _)

    GPUOperator(outputType, "ColorShift") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val a = _readTensorElement(img, _tensorElement)

      //val shift = _readTensorElement(colorShift, _tensorElement % 3)
      //_writeTensorElement(_out0, a + shift, _tensorElement)

      val i = _floatVar()
      i := _tensorElement % 3
      _if (_isequal((i),0f)) {
        val rShift = _readTensor(colorShift, 0)
        _writeTensorElement(_out0, a + rShift, _tensorElement)
      }
      _elseif (_isequal((i),1f)) {
        val gShift = _readTensor(colorShift, 1)
        _writeTensorElement(_out0, a + gShift, _tensorElement)
      }
      _else {
        //if (_isequal((_tensorElement % 3),2)) {
        val bShift = _readTensor(colorShift, 2)
        _writeTensorElement(_out0, a + bShift, _tensorElement)
      }

    }
  }

  /**
   * applyColorShiftPerImage GPU operator
   * gets 3 * batchSize shift values from sensor
   * adds unique 3-color shift to each image in the batch
   */


  def applyColorShiftPerImage(img: Field, batchSize: Int): Field = {
    require(img.fieldShape.dimensions == 2, "Requires a 2D field")
    val outputType = img.fieldType

    def nextValueRGB2:Option[Iterator[Float]] = {

      val resultArray = new Array[Float](3*batchSize)

      for (x <- 1 to batchSize) {

        val alpha1 = 0.1 * (rng.nextGaussian())
        val alpha2 = 0.1 * (rng.nextGaussian())
        val alpha3 = 0.1 * (rng.nextGaussian())

        val p1 = (-0.56771488, -0.58145678, -0.5827588)
        val p2 = (-0.72252667, 0.01267614, 0.69122683)
        val p3 = (0.3945314, -0.81347853, 0.42731446)

        val lambda1 = 0.204896523508
        val lambda2 = 0.0194841685264
        val lambda3 = 0.00456344555301

        val w1 = alpha1 * lambda1
        val w2 = alpha2 * lambda2
        val w3 = alpha3 * lambda3

        val rShift = (p1._1 * w1 + p2._1 * w2 + p3._1 * w3).toFloat
        val gShift = (p1._2 * w1 + p2._2 * w2 + p3._2 * w3).toFloat
        val bShift = (p1._3 * w1 + p2._3 * w2 + p3._3 * w3).toFloat

        val index = 3*(x-1)

        resultArray(index) = rShift
        resultArray(index+1) = gShift
        resultArray(index+2) = bShift

      }
      Option(resultArray.iterator)
    }

    val colorShift = new Sensor(Shape(3*batchSize), nextValueRGB2 _, resetHook _)

    GPUOperator(outputType, "ColorShiftPerImage") {
      _globalThreads(outputType.fieldShape, outputType.tensorShape)
      val a = _readTensorElement(img, _tensorElement)

      val i = _floatVar()
      i := _tensorElement % 3
      val j = _floatVar()
      j := _tensorElement % batchSize
      val k = _intVar()
      k := 3*j+i

      val shift = _readTensor(colorShift, k)
      _writeTensorElement(_out0, a + shift, _tensorElement)
    }
  }
}
