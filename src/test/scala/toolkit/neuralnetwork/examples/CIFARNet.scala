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

import java.io.File

import cogdebugger.CogDebuggerApp
import libcog._
import toolkit.neuralnetwork.Implicits._
import toolkit.neuralnetwork.examples.util.AveragePooling
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.layer.{BiasLayer, ConvolutionLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.policy.StandardLearningRule
import toolkit.neuralnetwork.source.{ByteDataSource, ByteLabelSource}
import toolkit.neuralnetwork.util.{CorrectCount, NormalizedLowPass}


object CIFARNet extends CogDebuggerApp(new ComputeGraph {
  val dir = new File(System.getProperty("user.home"), "cog/data/cifar10")
  val batchSize = 300
  val outputClasses = 10
  val LR = 0.01f
  val momentum = 0.9f
  val decay = 0.0005f

  val lr = StandardLearningRule(LR, momentum, decay)

  val data = ByteDataSource(new File(dir, "training_data.bin").toString, Shape(32, 32), 3, batchSize)
  val label = ByteLabelSource(new File(dir, "training_labels.bin").toString, 10, batchSize)

  val c1 = ConvolutionLayer(data, Shape(5, 5), 32, BorderZero, lr)
  val b1 = BiasLayer(c1, lr)
  val m1 = MaxPooling(b1, poolSize = 3, stride = 2)
  val r1 = ReLU(m1)

  val c2 = ConvolutionLayer(r1, Shape(5, 5), 32, BorderZero, lr)
  val b2 = BiasLayer(c2, lr)
  val r2 = ReLU(b2)
  val p2 = AveragePooling(r2, poolSize = 3, stride = 2)

  val c3 = ConvolutionLayer(p2, Shape(5, 5), 64, BorderZero, lr)
  val b3 = BiasLayer(c3, lr)
  val r3 = ReLU(b3)
  val p3 = AveragePooling(r3, poolSize = 3, stride = 2)

  val fc64 = FullyConnectedLayer(p3, 64, lr)
  val r64 = ReLU(fc64)

  val fc10 = FullyConnectedLayer(r64, 10, lr)
  val loss = CrossEntropySoftmax(fc10, label) / batchSize

  val correct = CorrectCount(fc10.forward, label.forward, batchSize, 0.01f) / batchSize
  val avgCorrect = NormalizedLowPass(correct, 0.01f)
  val avgLoss = NormalizedLowPass(loss.forward, 0.01f)

  loss.activateSGD()

  probe(data.forward)
  probe(label.forward)
  probe(loss.forward)
  probe(correct)
  probe(avgCorrect)
  probe(avgLoss)
})
