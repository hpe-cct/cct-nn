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

import cogio._
import libcog._


case class FCInit(inputLen: Int) extends WeightInitPolicy {
  override def initState(fieldShape: Shape, tensorShape: Shape): Field = {
    require(tensorShape.dimensions == 1)
    val stateType = new FieldType(fieldShape, tensorShape, Float32)
    val m = 1f
    val sigma = m / math.sqrt(fieldShape.points * inputLen).toFloat
    val dist = NormalDistribution(0f, sigma)
    val rng = new Random
    val dat = IndexedSeq.tabulate(fieldShape.points * tensorShape.points) { i => dist.distFunc(rng.nextFloat) }
    FieldState(stateType, dat.toVector).toField
  }
}
