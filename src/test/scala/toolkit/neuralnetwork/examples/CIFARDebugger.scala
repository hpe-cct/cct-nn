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

package toolkit.neuralnetwork.examples

import cogdebugger.CogDebuggerApp
import libcog._
import toolkit.neuralnetwork.examples.networks.CIFAR


object CIFARDebugger extends CogDebuggerApp(new ComputeGraph {
  val net = new CIFAR(useRandomData = false, learningEnabled = true, batchSize = 100)

  probe(net.data.forward)
  probe(net.label.forward)
  probe(net.loss.forward)
  probe(net.correct)
  probe(net.avgCorrect)
  probe(net.avgLoss)
})
