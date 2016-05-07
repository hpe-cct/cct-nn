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

import cogx.utilities.Random
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.source.RandomSource


object Dropout {
  def apply(input: DifferentiableField, enabled: Boolean = true, seed: Option[Long] = None): DifferentiableField = {
    if (enabled) {
      val shape = input.forward.fieldShape
      val len = input.forward.tensorShape(0) / input.batchSize

      // If we're given a seed, use it. If not, get one from the Cog seed factory.
      val _seed = Some(seed.getOrElse(Random.nextSeed))

      val dropoutGenerator = RandomSource(shape, len, input.batchSize, bits = 1, seed = _seed)

      input * dropoutGenerator
    } else {
      input
    }
  }
}
