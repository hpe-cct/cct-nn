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

package toolkit.neuralnetwork.examples.networks

import toolkit.neuralnetwork.{DifferentiableField, WeightStore}
import libcog._

trait Net {
  val correct: Field
  val loss: DifferentiableField
  val weights: WeightStore
}

object Net {
  def apply(netName: Symbol, useRandomData: Boolean, learningEnabled: Boolean, batchSize: Int,
            training: Boolean = true, weights: WeightStore = WeightStore()): Net = {
    netName match {
      case 'CIFAR => new CIFAR(useRandomData, learningEnabled, batchSize, training, weights)
      case 'SimpleConvNet => new SimpleConvNet(useRandomData, learningEnabled, batchSize, training, weights)
      case s => throw new IllegalArgumentException(s"unknown network $s")
    }
  }
}
