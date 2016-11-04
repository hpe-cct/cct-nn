package toolkit.neuralnetwork.function

import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.DifferentiableField.GradientPort


/**
  * Created by Michael Neary on 7/29/2016.
  */
case class Sigmoid(input: DifferentiableField) extends DifferentiableField{
  override val batchSize: Int = input.batchSize
  override val forward: Field = sigmoid(input.forward)
  override val inputs: Map[Symbol, GradientPort] = Map('input -> GradientPort(input, jacobian, jacobian))

  def sigmoid(f: Field): Field = 1f / (1f + exp(-f))

  //computes the jacobian of the sigmoid function
  def jacobian(dx: Field): Field = {
    sigmoid(input.forward) * (1f - sigmoid(input.forward)) * dx
  }
}
