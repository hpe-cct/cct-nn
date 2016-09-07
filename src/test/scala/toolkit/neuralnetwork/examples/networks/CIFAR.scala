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
import toolkit.neuralnetwork.{DifferentiableField, WeightStore}
import toolkit.neuralnetwork.examples.util.AveragePooling
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.layer.{BiasLayer, ConvolutionLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.policy.{GaussianInit, StandardLearningRule}
import toolkit.neuralnetwork.source.{ByteDataSource, ByteLabelSource, RandomSource}
import toolkit.neuralnetwork.util.{CorrectCount, NormalizedLowPass}


class CIFAR(useRandomData: Boolean, learningEnabled: Boolean, batchSize: Int,
            training: Boolean = true, val weights: WeightStore = WeightStore()) extends Net {
  val LR = 0.001f
  val momentum = 0.9f
  val decay = 0.004f

  val lr = StandardLearningRule(LR, momentum, decay)
  // Bias weights have twice the base learning rate
  val biasLr = StandardLearningRule(LR * 2f, momentum, decay)

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

      probe(mean.forward)

      (Bias(data, mean), label)
  }

  val c1 = ConvolutionLayer(data, Shape(5, 5), 32, BorderZero, lr, initPolicy = GaussianInit(0.0001f), weightBinding = weights.bind('c1))
  val b1 = BiasLayer(c1, biasLr, weightBinding = weights.bind('b1))
  val m1 = MaxPooling(b1, poolSize = 3, stride = 2)
  val r1 = ReLU(m1)

  val c2 = ConvolutionLayer(r1, Shape(5, 5), 32, BorderZero, lr, initPolicy = GaussianInit(0.01f), weightBinding = weights.bind('c2))
  val b2 = BiasLayer(c2, biasLr, weightBinding = weights.bind('b2))
  val r2 = ReLU(b2)
  val p2 = AveragePooling(r2, poolSize = 3, stride = 2)

  val c3 = ConvolutionLayer(p2, Shape(5, 5), 64, BorderZero, lr, initPolicy = GaussianInit(0.01f), weightBinding = weights.bind('c3))
  val b3 = BiasLayer(c3, biasLr, weightBinding = weights.bind('b3))
  val r3 = ReLU(b3)
  val p3 = AveragePooling(r3, poolSize = 3, stride = 2)

  val fc64 = FullyConnectedLayer(p3, 64, lr, initPolicy = GaussianInit(0.1f), weightBinding = weights.bind('fc64))
  val bfc64 = BiasLayer(fc64, biasLr, weightBinding = weights.bind('bfc64))
  val r64 = ReLU(bfc64)

  val fc10 = FullyConnectedLayer(r64, 10, lr, initPolicy = GaussianInit(0.1f), weightBinding = weights.bind('fc10))
  val bfc10 = BiasLayer(fc10, biasLr, weightBinding = weights.bind('bfc10))
  val loss = CrossEntropySoftmax(bfc10, label) / batchSize

  val correct = CorrectCount(bfc10.forward, label.forward, batchSize, 0.01f) / batchSize
  val avgCorrect = NormalizedLowPass(correct, 0.01f)
  val avgLoss = NormalizedLowPass(loss.forward, 0.01f)

  if (learningEnabled) {
    loss.activateSGD()
    probe(loss.forward)
  } else {
    probe(bfc10.forward)
  }
}
