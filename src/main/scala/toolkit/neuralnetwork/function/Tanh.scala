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

/** The function a*tanh(b*x) applied at each point
  *
  * @author Matthew Pickett
  * @param input the input signal
  * @param a     scale parameter
  * @param b     gain parameter
  */
case class Tanh(input: DifferentiableField, a: Float = 1.7159f, b: Float = 0.6667f) extends DifferentiableField {
  override val batchSize: Int = input.batchSize
  override val forward: libcog.Field = a * tanh(b * input.forward)
  override val inputs: Map[Symbol, GradientPort] =
    Map('input -> GradientPort(input, jacobian, jacobian))

  private def jacobian(dx: Field): Field = {
    a * b * (1f - sq(tanh(b * input.forward))) * dx
  }
}
