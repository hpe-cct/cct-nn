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

package toolkit.neuralnetwork.function

import libcog._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.DifferentiableField.GradientPort


/** The downsample operator which samples every `factor` input element to produce a reduced size output.
  * The outputSize = ceil(inputSize / factor).
  *
  * Decided not to expose the "phase" parameter that exists with Cog's core downsample/upsample operators.
  * There are some subtleties present in the Cog API, namely phase <= inSize-1 % factor.  Put a different way,
  * the Cog core's downsample() will result in a field of size ceil(inSize/factor).  The specification of a
  * non-zero phase should not disturb that. (inSize, factor, phase) = (5, 3, 2) is such a problem case where
  * one might assume an output of size 1, but Cog will generate a size of 2.
  *
  * @author Dick Carter
  * @param input  input signal
  * @param factor the factor by which the output is reduced
  */
class Downsample private[Downsample] (input: DifferentiableField, factor: Int) extends DifferentiableField {
  private val in = (input.forward, input.batchSize)

  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = _forward(in)._1
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, dx => jacobian(dx, in), grad => jacobianAdjoint(grad, in)))

  /** The Downsample operator's forward function. */
  private def _forward(in: (Field, Int)): (Field, Int) = {
    val x = in._1
    val batchSize = in._2
    val outField = downsample(x, factor)
    (outField, batchSize)
  }

  /** The Downsample operator's jacobian function- used to validate the jacobianAdjoint. */
  private def jacobian(dx: Field, in: (Field, Int)): Field = {
    downsample(dx, factor)
  }

  /** The Downsample operator's jacobian adjoint function. */
  private def jacobianAdjoint(grad: Field, in: (Field, Int)): Field = {
    val x = in._1
    val untrimmed = upsample(grad, factor)
    if (untrimmed.fieldShape == x.fieldShape)
      untrimmed
    else
      trim(untrimmed, x.fieldShape)
  }

  // If you add/remove constructor parameters, you should alter the toString() implementation. */
  /** A string description of the instance in the "case class" style. */
  override def toString = this.getClass.getName +
    (input, factor)
}

/** Factory object- eliminates clutter of 'new' operator. */
object Downsample {
  /** The downsample operator which samples every `factor` input element to produce a reduced size output.
    * The outputSize = ceil(inputSize / factor).
    *
    * Decided not to expose the "phase" parameter that exists with Cog's core downsample/upsample operators.
    * There are some subtleties present in the Cog API, namely phase <= inSize-1 % factor.  Put a different way,
    * the Cog core's downsample() will result in a field of size ceil(inSize/factor).  The specification of a
    * non-zero phase should not disturb that. (inSize, factor, phase) = (5, 3, 2) is such a problem case where
    * one might assume an output of size 1, but Cog will generate a size of 2.
    *
    * @param input  input signal
    * @param factor the factor by which the output is reduced
    */
  def apply(input: DifferentiableField, factor: Int = 2) =
    new Downsample(input, factor)
}
