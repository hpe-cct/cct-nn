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

import libcog._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.Implicits._
import toolkit.neuralnetwork.examples.util.{DataAugmentation, VectorMeanSquares}
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.layer.{ConvolutionLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.policy.{Space, StandardLearningRule}
import toolkit.neuralnetwork.source.{ByteDataSource, FloatLabelSource, RandomSource}
import toolkit.neuralnetwork.util.{CorrectCount, NormalizedLowPass}


/** Neural network model from Alex Krishevsky, et.al.,  See:
  *
  * https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  *
  * @param batchSize The number of training images processed per simulation tick.
  * @param enableNormalization Apply normalization to the first two layers.
  * @param useRandomData Don't feed in real images, just generate random data (for performance studies).
  *
  * @author Ben Chandler and Dick Carter
  */
class AlexNet(batchSize: Int, enableNormalization: Boolean, useRandomData: Boolean) {
  // Other Network parameters:
  // weight update rule and parameters
  val lr = StandardLearningRule(0.01f, 0.9f, 0.0005f)
  // parameters for AlexNet normalization layers
  val (k, alpha, beta, windowSize) = (2.0f, 1e-4f, 0.75f, 5)

  // Data parameters:
  // paths to the mean image file, training images, and training labels for real data option
  val imagenetRoot = "/fdata/scratch/imagenet/"
  val meanImageFile = imagenetRoot + "TrainingMeanImage1.bin"
  val trainingImages = imagenetRoot + "TrainingImages1.bin"
  val labelFile = imagenetRoot + "TrainingLabels1.bin"

  // Tuning parameters:
  // use Maxwell-optimized convolution? set to false on Kepler or prior architectures.
  Convolution.tuneForNvidiaMaxwell = true

  def normalize(in: DifferentiableField): DifferentiableField = {
    if (!enableNormalization) {
      in
    } else {
      in * AplusBXtoN(VectorMeanSquares(in, windowSize, BorderCyclic), a = k, b = 5 * alpha, n = -beta)
    }
  }

  val data: DifferentiableField = if (useRandomData) {
    RandomSource(Shape(230, 230), 3, batchSize)
  } else {
    val meanImage = DataAugmentation.loadOffsetVector(meanImageFile)

    def meanImageAsVectorField: VectorField = {
      VectorField(meanImage.length, meanImage(0).length, (i, j) => meanImage(i)(j))
    }

    // Load 256x256x3 samples from disk
    val raw = ByteDataSource(trainingImages, Shape(256, 256), 3, batchSize)
    // Subtract the mean image and apply a random crop and reflection
    val pre1 = DataAugmentation.subtractCropReflect2(raw.forward, meanImageAsVectorField, Shape(230, 230))
    // Apply the AlexNet color shift data augmentation
    val pre2 = DifferentiableField(DataAugmentation.applyColorShiftPerImage(pre1, batchSize), batchSize)

    pre2
  }

  val label: DifferentiableField = if (useRandomData) {
    RandomSource(Shape(), 1000, batchSize)
  } else {
    FloatLabelSource(labelFile, 1000, batchSize)
  }

  val c1 = ConvolutionLayer(data, Shape(11, 11), 96, BorderValid, lr, stride = 4, impl = Space)
  val r1 = ReLU(c1)
  val n1 = normalize(r1)
  val p1 = MaxPooling(n1, poolSize = 3, stride = 2)

  val c2 = ConvolutionLayer(p1, Shape(5, 5), 256, BorderZero, lr)
  val r2 = ReLU(c2)
  val n2 = normalize(r2)
  val p2 = MaxPooling(n2, poolSize = 3, stride = 2)

  val c3 = ConvolutionLayer(p2, Shape(3, 3), 384, BorderZero, lr)
  val r3 = ReLU(c3)

  val c4 = ConvolutionLayer(r3, Shape(3, 3), 384, BorderZero, lr)
  val r4 = ReLU(c4)

  val c5 = ConvolutionLayer(r4, Shape(3, 3), 256, BorderZero, lr)
  val r5 = ReLU(c5)
  val p5 = MaxPooling(r5, poolSize = 3, stride = 2)

  val fc6 = FullyConnectedLayer(p5, 4096, lr)
  val d6 = Dropout(fc6)
  val r6 = ReLU(d6)

  val fc7 = FullyConnectedLayer(r6, 4096, lr)
  val d7 = Dropout(fc7)
  val r7 = ReLU(d7)

  val fc8 = FullyConnectedLayer(r7, 1000, lr)

  val loss = CrossEntropySoftmax(fc8, label) / batchSize

  loss.activateSGD()

  val correct = CorrectCount(fc8.forward, label.forward, batchSize, 0.01f) / batchSize
  val avgCorrect = NormalizedLowPass(correct, 0.001f)
  val avgLoss = NormalizedLowPass(loss.forward, 0.001f)

  probe(data.forward)
  probe(label.forward)
  probe(loss.forward)
  probe(correct)
  probe(avgCorrect)
  probe(avgLoss)
}
