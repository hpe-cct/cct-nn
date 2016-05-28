package toolkit.neuralnetwork.examples

import libcog._
import cogdebugger._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.function._
import toolkit.neuralnetwork.layer.{ConvolutionLayer, FullyConnectedLayer}
import toolkit.neuralnetwork.policy.{Space, StandardLearningRule}
import toolkit.neuralnetwork.source.RandomSource
import toolkit.neuralnetwork.Implicits._


object AlexNet extends CogDebuggerApp(new ComputeGraph {
  val batchSize = 128
  val lr = StandardLearningRule(0.01f, 0.9f, 0.0005f)

  Convolution.tuneForNvidiaMaxwell = true

  val data = RandomSource(Shape(230, 230), 3, batchSize)
  val label = RandomSource(Shape(), 1000, batchSize)

  // AlexNet normalization isn't ported yet. This is just a pass-through.
  def normalize(input: DifferentiableField): DifferentiableField = input

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
})
