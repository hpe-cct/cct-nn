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

import cogio.fieldstate.FieldState
import libcog._


case class GaussianInit(sigma: Float) extends WeightInitPolicy {
  require(sigma > 0f, s"sigma must be positive and non-zero, got $sigma")

  override def initState(fieldShape: Shape, tensorShape: Shape): Field = {
    val stateType = new FieldType(fieldShape, tensorShape, Float32)
    val statePoints = fieldShape.points * tensorShape.points

    val dist = NormalDistribution(0f, sigma)
    val rng = new Random
    val dat = IndexedSeq.tabulate(statePoints) { i => dist.distFunc(rng.nextFloat) }
    FieldState(stateType, dat.toVector).toField
  }
}
