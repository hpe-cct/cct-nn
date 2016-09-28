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

package toolkit.neuralnetwork

import libcog._
import cogio.FieldState

import scala.collection.mutable


trait WeightStore {
  private[neuralnetwork] val bindings = mutable.Map[Symbol, Option[Field]]()

  def bind(symbol: Symbol): WeightBinding

  def snapshot(source: ComputeGraph): Map[Symbol, FieldState] = {
    // First, check that all the bound state fields registered correctly
    require(bindings.values.forall(_.isDefined), s"weight field ${bindings.find(p => p._2.isEmpty).get._1} failed to register")

    // Iterate through the bindings, reading the field from each
    bindings.mapValues(v => {
      FieldState.read(source.read(v.get))
    }).toMap
  }
}

object WeightStore {
  // Build a weight store with no pre-existing weight state
  def apply(): WeightStore = new WeightStore {
    def bind(symbol: Symbol): WeightBinding = {
      require(!bindings.contains(symbol), s"$symbol is already bound")
      bindings(symbol) = None

      new WeightBinding {
        def initialWeights: Option[Field] = None

        def register(weights: Field): Unit = {
          bindings(symbol) = Some(weights)
        }
      }
    }
  }

  def restoreFromSnapshot(snapshot: Map[Symbol, FieldState]): WeightStore = {
    val _snapshot = snapshot

    new WeightStore {
      private val initState = _snapshot

      def bind(symbol: Symbol): WeightBinding = {
        require(!bindings.contains(symbol), s"$symbol is already bound")
        bindings(symbol) = None

        new WeightBinding {
          def initialWeights: Option[Field] = initState.get(symbol).map(fs => {
            val field = fs.toField
            register(field)
            field
          })

          def register(weights: Field): Unit = {
            require(bindings.contains(symbol) && bindings(symbol).isEmpty)
            bindings(symbol) = Some(weights)
          }
        }
      }
    }
  }
}