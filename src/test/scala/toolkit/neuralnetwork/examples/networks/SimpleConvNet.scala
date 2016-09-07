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

import java.io.File

import cogio.fieldstate.FieldState
import libcog._
import toolkit.neuralnetwork.Implicits._
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.layer.{BiasLayer, ConvolutionLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.policy.StandardLearningRule
import toolkit.neuralnetwork.source.{ByteDataSource, ByteLabelSource, RandomSource}
import toolkit.neuralnetwork.util.{CorrectCount, NormalizedLowPass}
import toolkit.neuralnetwork.{DifferentiableField, WeightStore}


class SimpleConvNet(useRandomData: Boolean, learningEnabled: Boolean, batchSize: Int,
                    training: Boolean = true, val weights: WeightStore = WeightStore()) extends Net {
  val LR = 0.01f
  val momentum = 0.9f
  val decay = 0.0005f

  val lr = StandardLearningRule(LR, momentum, decay)

  val (data, label) = if (useRandomData) {
    (RandomSource(Shape(32, 32), 3, batchSize), RandomSource(Shape(), 10, batchSize))
  } else {
      val prefix = if (training) {
        "training"
      } else {
        "testing"
      }
      val dir = new File(System.getProperty("user.home"), "cog/data/cifar10")
      val data = ByteDataSource(new File(dir, s"${prefix}_data.bin").toString, Shape(32, 32), 3, batchSize)
      val mean = DifferentiableField(FieldState.loadFromFile(new File("cifar_mean.field")).toField, 1) * -1f
      val label = ByteLabelSource(new File(dir, s"${prefix}_labels.bin").toString, 10, batchSize)

      (Bias(data, mean), label)
  }

  val c1 = ConvolutionLayer(data, Shape(5, 5), 32, BorderValid, lr, weightBinding = weights.bind('c1))
  val b1 = BiasLayer(c1, lr, weightBinding = weights.bind('b1))
  val m1 = MaxPooling(b1)
  val r1 = ReLU(m1)

  val c2 = ConvolutionLayer(r1, Shape(5, 5), 32, BorderValid, lr, weightBinding = weights.bind('c2))
  val b2 = BiasLayer(c2, lr, weightBinding = weights.bind('b2))
  val m2 = MaxPooling(b2)
  val r2 = ReLU(m2)

  val fc3 = FullyConnectedLayer(r2, 256, lr, weightBinding = weights.bind('fc3))
  val b3 = BiasLayer(fc3, lr, weightBinding = weights.bind('b3))
  val r3 = Tanh(b3)

  val fc4 = FullyConnectedLayer(r3, 10, lr, weightBinding = weights.bind('fc4))
  val b4 = BiasLayer(fc4, lr, weightBinding = weights.bind('b4))
  val loss = CrossEntropySoftmax(b4, label) / batchSize

  val correct = CorrectCount(b4.forward, label.forward, batchSize, 0.01f) / batchSize
  val avgCorrect = NormalizedLowPass(correct, 0.001f)
  val avgLoss = NormalizedLowPass(loss.forward, 0.001f)

  if (learningEnabled) {
    loss.activateSGD()
    probe(loss.forward)
  } else {
    probe(b4.forward)
  }
}
