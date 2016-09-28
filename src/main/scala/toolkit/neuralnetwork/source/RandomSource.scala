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


class RandomSource private[RandomSource] (fieldShape: Shape, vectorLen: Int, override val batchSize: Int,
                        bits: Int, seed: Option[Long], override val gradientConsumer: Boolean) extends DifferentiableField {

  override val forward: libcog.Field = {
    val rng = seed match {
      case Some(s) => new Random(s)
      case None => new Random()
    }

    // The Cog random operator only works for fields with > 64 points. Ensure that the generator field is large
    // enough, and take a subset of points for the output field if it needs to be expanded. This is done by
    // extending the field in the vector dimension but cutting it off afterwards.
    val desiredPoints = fieldShape.points * vectorLen * batchSize
    val minPoints = 64
    val extraPoints = minPoints - desiredPoints
    val extraElems = if (extraPoints <= 0) 0 else extraPoints / fieldShape.points + 1

    //Define the vector generation function which initializes vectors randomly at the desired precision
    val nextElem = () => {
      val maxVal = math.pow(2, bits).toInt
      val rand = rng.nextInt(maxVal)
      rand.toFloat / (maxVal - 1)
    }
    val nextVec = () => Vector(vectorLen * batchSize + extraElems, (i: Int) => nextElem())

    //Define the core generator field
    val field = fieldShape.dimensions match {
      case 0 => VectorField(nextVec())
      case 1 => VectorField(fieldShape(0), (c: Int) => nextVec())
      case 2 => VectorField(fieldShape(0), fieldShape(1), (r: Int, c: Int) => nextVec())
      case 3 => VectorField(fieldShape(0), fieldShape(1), fieldShape(2), (l: Int, r: Int, c: Int) => nextVec())
      case _ => throw new RuntimeException("Bad dimensionality")
    }

    field <== random(field, bits)

    //crop the field to the desired field if the desired number of points is less than 64
    val indices = VectorField(Vector(vectorLen * batchSize, (i: Int) => i))
    val croppedField =
      if (desiredPoints == 1) reduceSum(vectorElements(field, indices))
      else vectorElements(field, indices)

    croppedField
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (fieldShape, vectorLen, batchSize, bits, seed, gradientConsumer)
}

object RandomSource {
  def apply(fieldShape: Shape, vectorLen: Int, batchSize: Int,
            bits: Int = 12, seed: Option[Long] = None, gradientConsumer: Boolean = false) =
    new RandomSource(fieldShape, vectorLen, batchSize, bits, seed, gradientConsumer)
}
