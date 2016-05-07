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

package toolkit.neuralnetwork.policy

import org.apache.commons.math3.special.Erf

/** Case classes for picking randomly from different types of distributions
  */
trait Distribution extends Serializable {
  def distFunc(uniform: Float): Float
}

/*pick a number between 0 and 1, multiply by scale, subtract offset*/
case class UniformDistribution(scale: Float, offset: Float) extends Distribution {
  def distFunc(uniform: Float) =
    uniform * scale - offset
}

/*pick from a normal distribution parameterized by mu and sigma*/
case class NormalDistribution(mu: Float, sigma: Float) extends Distribution {
  def distFunc(uniform: Float) =
    (mu - math.sqrt(2.0) * sigma * Erf.erfcInv(2.0 * uniform + 1e-9f)).toFloat
}