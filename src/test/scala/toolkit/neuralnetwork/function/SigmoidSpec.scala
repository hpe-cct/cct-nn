package toolkit.neuralnetwork.function

import toolkit.neuralnetwork.{ComputeTests, DifferentiableField, UnitSpec}

/**
  * Created by Michael Neary on 8/1/2016.
  */
class SigmoidSpec extends UnitSpec with ComputeTests {
  val fn = {
    (s: Seq[DifferentiableField]) => Sigmoid(s.head)
  }

  "The Sigmoid operator" should "support 0D input" in {
    val inputShapes = Seq(Shape())
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 1D input" in {
    val inputShapes = Seq(Shape(15))
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

  it should "support 2D input" in {
    val inputShapes = Seq(Shape(15, 29))
    val inputLens = Seq(73)
    val batchSizes = Seq(33)

    jacobian(fn, inputShapes, inputLens, batchSizes)
    jacobianAdjoint(fn, inputShapes, inputLens, batchSizes)
  }

}
