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
