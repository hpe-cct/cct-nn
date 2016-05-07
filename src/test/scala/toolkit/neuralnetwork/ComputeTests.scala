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
import org.scalatest.Matchers
import toolkit.neuralnetwork.util.{Norm, NormalizedLowPass}

/** Trait for establishing minimum testing standards for `LayerFunction`s */
trait ComputeTests {
  this: Matchers =>
  // This is the only numerical setting in the current implementation. The new test code uses the central difference,
  // which ought to be more stable than either the forward or backward versions. The value of 'dx' needs to
  // be well above the single-precision machine epsilon to prevent excessive round-off error, but still tiny
  // to avoid excessive integration error.
  //
  // A DifferentiableField function passes the tests below if the mean L1 error is below the value of dx.
  def dx = 0.001f

  // Step parameters. A test will run until either maxPoints field points have been tested or
  // timeBudget milliseconds have elapsed *after* the first step. The first step is excluded from
  // the time budget to prevent compilation time from washing out step time. If the time budget is
  // insufficient to reach minPoints test points, the graph will continue stepping until at least
  // minPoints test points have been covered.
  def minPoints = 1000L
  def maxPoints = 1000000L
  def timeBudget = 500L

  private def generateMasks(length: Int): Seq[Seq[Boolean]] = {
    // Mask with all input ports active
    val totalMask = Seq.tabulate(length) {_ => true}

    // Mask with exactly one input port active
    val partialMasks = Seq.tabulate(length) {
      i => Seq.tabulate(length) {
        j => i == j
      }
    }

    // If there are two or more ports, test both the total error and per-port error
    // With exactly one port, the per-port error equals the total error
    if (partialMasks.length >= 2) {
      Seq(totalMask) ++ partialMasks
    } else {
      Seq(totalMask)
    }
  }

  def jacobian(fn: Seq[DifferentiableField] => DifferentiableField,
               inputShapes: Seq[Shape], inputLens: Seq[Int], batchSizes: Seq[Int],
               positiveXOnly: Boolean = false, toleranceMultiplier: Float = 1f) = {
    require(inputShapes.length == inputLens.length, "all argument lists must have the same length")
    require(inputShapes.length == batchSizes.length, "all argument lists must have the same length")
    require(toleranceMultiplier > 0f, "tolerance multiplier must be greater than zero")

    println("Attempting jacobian test...")

    val cg = new ComputeGraph {
      def offset(i: DifferentiableField) = if (positiveXOnly) {
        i + 5f
      } else {
        i - 0.5f
      }

      def finiteDifferenceErr(mask: Seq[Boolean], f: Seq[DifferentiableField] => DifferentiableField): (Field, Field, Field) = {
        // Define the base test point. This is a random source of the user-specified dimensions.
        // It's marked as a gradient consumer if the corresponding `mask` element is true.
        val x = Seq.tabulate(inputLens.length) {
          i => offset(source.RandomSource(inputShapes(i), inputLens(i), batchSizes(i),
            gradientConsumer = mask(i), seed = Some(Random.nextSeed)))
        }

        // This function uses the central limit method. These are the positive and negative deltas from the
        // base test point. Only add or subtract `dx` if the corresponding port (indicated by `mask`) is active.
        val xPositive = Seq.tabulate(x.length) {
          i => if (mask(i)) {
            x(i) + dx
          } else {
            x(i)
          }
        }

        val xNegative = Seq.tabulate(x.length) {
          i => if (mask(i)) {
            x(i) - dx
          } else {
            x(i)
          }
        }

        // Calculate the numeric and symbolic total derivatives, then return the
        // average L1 error between the two.
        val numeric = (f(xPositive).forward - f(xNegative).forward) / (2f * dx)
        val symbolic = f(x).totalDerivative()

        val err = Norm.L1(numeric, symbolic)
        (numeric, symbolic, NormalizedLowPass(err, 0.0001f))
      }

      val masks = generateMasks(inputShapes.length)

      // Generate this, but don't probe it. Necessary for determining the output size.
      val (numeric, symbolic, totalErr) = finiteDifferenceErr(masks.head, fn)

      // Probe the entire sequence of errors, total error first.
      val errs = masks.map(m => {
        val err = finiteDifferenceErr(m, fn)._3
        probe(err)
        err
      })
    }

    // The number of test points per step is counted at the *output*, not the inputs
    val pointsPerStep: Long = cg.numeric.fieldShape.points * cg.numeric.tensorShape.points
    var completedPoints = 0L

    val err = cg.withRelease {
      // Step once to trigger compilation
      cg.step
      completedPoints += pointsPerStep
      val startTime = System.nanoTime()

      // Keep stepping until at least the minimum number of points are covered and either the time budget or
      // maximum number of points are reached.
      while (completedPoints < minPoints ||
        (completedPoints < maxPoints && (System.nanoTime() - startTime) < timeBudget * 1000000L)) {
        cg.step
        completedPoints += pointsPerStep
      }

      val errs = cg.errs.map(field => {
        val reader = cg.read(field).asInstanceOf[ScalarFieldReader]
        reader.read()
      })

      // Total error is the error with all input ports active
      val totalErr = errs.head
      // Port-wise error is a list of errors with only the corresponding input port active
      val portwiseErrs = errs.tail

      all (portwiseErrs) should be < dx * toleranceMultiplier
      totalErr should be < dx * toleranceMultiplier
    }

    println("Jacobian test passed.")
  }

  def jacobianAdjoint(fn: Seq[DifferentiableField] => DifferentiableField,
                      inputShapes: Seq[Shape], inputLens: Seq[Int], batchSizes: Seq[Int],
                      positiveXOnly: Boolean = false, toleranceMultiplier: Float = 1f) = {
    require(inputShapes.length == inputLens.length, "all argument lists must have the same length")
    require(inputShapes.length == batchSizes.length, "all argument lists must have the same length")
    require(toleranceMultiplier > 0f, "tolerance multiplier must be greater than zero")

    println("Attempting jacobian adjoint test...")

    def offset(i: DifferentiableField) = if (positiveXOnly) {
      i + 5f
    } else {
      i - 0.5f
    }

    val cg = new ComputeGraph {
      def adjointErr(mask: Seq[Boolean]): (Field, Field) = {
        // Pick a random starting points
        val b = Seq.tabulate(inputShapes.length) {
          i => offset(source.RandomSource(inputShapes(i), inputLens(i),
            batchSizes(i), seed = Some(Random.nextSeed)))
        }

        // And random inbound gradients
        val x = Seq.tabulate(inputShapes.length) {
          i => offset(source.RandomSource(inputShapes(i), inputLens(i),
            batchSizes(i), seed = Some(Random.nextSeed)))
        }

        // Build forward testing sources using random forward and forward gradient information
        // Use the mask to determine which input ports are active
        val forwardInput = Seq.tabulate(b.length) {
          i => AdjointTestSource(b(i), x(i), consumer = mask(i))
        }

        // At point `b`, propagate gradient `x` forward (wrapped in forwardInput)
        val forward = fn(forwardInput)
        val totalDerivative = forward.totalDerivative()

        // Generate a random top-down signal with dimensions matching `forward`
        val shape = forward.forward.fieldShape
        val batchSize = forward.batchSize
        val y = forward.forward.tensorOrder match {
          case 0 if batchSize == 1 =>
            // The forward pass produces a ScalarField and the batch size is one. Produce a random VectorField
            // with a single plane, then collapse that down to a ScalarField.
            val base = source.RandomSource(shape, 1, batchSize, seed = Some(Random.nextSeed)) - 0.5f
            new DifferentiableField {
              override val batchSize: Int = 1
              override val forward: libcog.Field = reduceSum(base.forward)
            }

          case 1 =>
            // The forward pass produces a VectorField. Make sure the batch size and number of planes are
            // consistent, then produce an appropriate random VectorField
            val vectorLength = forward.forward.tensorShape(0)
            require(vectorLength % batchSize == 0, s"number of forward planes ($vectorLength) must be an integer multiple of the batch size ($batchSize)")
            val planes = vectorLength / batchSize
            source.RandomSource(shape, planes, batchSize, seed = Some(Random.nextSeed)) - 0.5f

          case _ => throw new RuntimeException(s"illegal forward type (${forward.forward.fieldType}) given batch size ($batchSize)")
        }

        // Enable SGD and pull off the resulting adjoint at the input
        // Don't invoke callbacks to prevent user code from updating any forward state fields.
        forward.activateSGD(initField = y.forward, invokeCallbacks = false)
        val adjoint = forwardInput.map(_.backward)

        // The forward total derivative dotted with `y` should be ~= the backward stream dotted with `x`
        val forwardErr = fieldReduceSum(dot(totalDerivative, y.forward))
        // Reducing the N adjoint/x.forward dot product pairs to a single scalar
        // Note that the adjoint field will not exist for an inactive port. These are ignored.
        val backwardErr = adjoint.zip(x)
          .filter(_._1.isDefined) // Filter out inactive input ports
          .map(p => fieldReduceSum(dot(p._1.get, p._2.forward))) // Compute the dot product at each active port
          .reduce(_ + _) // Sum up the dot products across all active ports
        val err = NormalizedLowPass(Norm.L1(forwardErr, backwardErr), 0.0001f)

        (forward.forward, err)
      }

      val masks = generateMasks(inputLens.length)

      // Generate this, but don't probe it. Used in shape computations below.
      val (forward, totalErr) = adjointErr(masks.head)

      // Generate and probe error values for each mask permutation.
      val errs = masks.map(m => {
        val err = adjointErr(m)._2
        probe(err, s"mask $m")
        err
      })
    }

    val pointsPerStep: Long = cg.forward.fieldShape.points * cg.forward.tensorShape.points
    var completedPoints = 0L

    val err = cg.withRelease {
      // Step once to trigger compilation
      cg.step
      completedPoints += pointsPerStep
      val startTime = System.nanoTime()

      // Keep stepping until at least the minimum number of points are covered and either the time budget or
      // maximum number of points are reached.
      while (completedPoints < minPoints ||
        (completedPoints < maxPoints && (System.nanoTime() - startTime) < timeBudget * 1000000L)) {
        cg.step
        completedPoints += pointsPerStep
      }

      val errs = cg.errs.map(field => {
        val reader = cg.read(field).asInstanceOf[ScalarFieldReader]
        reader.read()
      })

      // Total error is the error with all input ports active
      val totalErr = errs.head
      // Port-wise error is a list of errors with only the corresponding input port active
      val portwiseErrs = errs.tail

      all (portwiseErrs) should be < dx * toleranceMultiplier
      totalErr should be < dx * toleranceMultiplier
    }

    println("Jacobian adjoint test passed.")
  }
}
