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

package toolkit.neuralnetwork.function

import libcog._
import toolkit.neuralnetwork.DifferentiableField
import toolkit.neuralnetwork.DifferentiableField.GradientPort
import toolkit.neuralnetwork.function.BasicOps._


trait BasicOps {
  def add(left: DifferentiableField, right: DifferentiableField): DifferentiableField = new Add(left, right)

  def add(input: DifferentiableField, c: Float): DifferentiableField = new AddConstant(input, c)

  def subtract(left: DifferentiableField, right: DifferentiableField): DifferentiableField = new Subtract(left, right)

  def subtract(input: DifferentiableField, c: Float): DifferentiableField = new SubtractConstant(input, c)

  def multiply(left: DifferentiableField, right: DifferentiableField): DifferentiableField = new Multiply(left, right)

  def multiply(input: DifferentiableField, c: Float): DifferentiableField = new MultiplyByConstant(input, c)

  def divide(left: DifferentiableField, right: DifferentiableField): DifferentiableField = left * pow(right, -1)

  def divide(input: DifferentiableField, c: Float): DifferentiableField = new DivideByConstant(input, c)

  /** Raise a node to a fixed power.  Cog has two pow() function signatures corresponding
    * to both integer and non-integer powers.  The integer case is detected here
    * and special-cased (instead of having a separate PowN node for this).
    *
    * If the power `n` is anything other than a positive integer, make sure the inputs
    * are always positive or NaNs will result.
    *
    * @author Matthew Pickett and Dick Carter
    * @param input the input signal
    * @param n     the power to raise the input to
    */
  def pow(input: DifferentiableField, n: Float): DifferentiableField = new Pow(input, n)
}

private[neuralnetwork] object BasicOps {

  // Since construction is limited to the uses above, we omit the Factory objects for these classes.

  class Add(left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
    require(left.batchSize == right.batchSize,
      s"batch sizes must match (got ${left.batchSize} and ${right.batchSize})")
    require(left.forward.fieldType == right.forward.fieldType,
      s"field types must match (got ${left.forward.fieldType} and ${right.forward.fieldType})")

    override val batchSize: Int = left.batchSize
    override val forward: libcog.Field = left.forward + right.forward
    override val inputs: Map[Symbol, GradientPort] = Map(
      'left -> GradientPort(left, dx => dx, grad => grad),
      'right -> GradientPort(right, dx => dx, grad => grad))
    override def toString = this.getClass.getName + (left, right)
  }

  class AddConstant(input: DifferentiableField, c: Float) extends DifferentiableField {
    override val batchSize: Int = input.batchSize
    override val forward: libcog.Field = input.forward + c
    override val inputs: Map[Symbol, GradientPort] =
      Map('input -> GradientPort(input, dx => dx, grad => grad))
    override def toString = this.getClass.getName + (input, c)
  }

  class Subtract(left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
    require(left.batchSize == right.batchSize,
      s"batch sizes must match (got ${left.batchSize} and ${right.batchSize})")
    require(left.forward.fieldType == right.forward.fieldType,
      s"field types must match (got ${left.forward.fieldType} and ${right.forward.fieldType})")

    override val batchSize: Int = left.batchSize
    override val forward: libcog.Field = left.forward - right.forward
    override val inputs: Map[Symbol, GradientPort] = Map(
      'left -> GradientPort(left, dx => dx, grad => grad),
      'right -> GradientPort(right, dx => dx * -1f, grad => grad * -1f))
    override def toString = this.getClass.getName + (left, right)
  }

  class SubtractConstant(input: DifferentiableField, c: Float) extends DifferentiableField {
    override val batchSize: Int = input.batchSize
    override val forward: libcog.Field = input.forward - c
    override val inputs: Map[Symbol, GradientPort] =
      Map('input -> GradientPort(input, dx => dx, grad => grad))
    override def toString = this.getClass.getName + (input, c)
  }

  class Multiply(left: DifferentiableField, right: DifferentiableField) extends DifferentiableField {
    require(left.batchSize == right.batchSize,
      s"batch sizes must match (got ${left.batchSize} and ${right.batchSize})")
    require(left.forward.fieldType == right.forward.fieldType,
      s"field types must match (got ${left.forward.fieldType} and ${right.forward.fieldType})")

    override val batchSize: Int = left.batchSize
    override val forward: libcog.Field = left.forward * right.forward
    override val inputs: Map[Symbol, GradientPort] = Map(
      'left -> GradientPort(left, dx => dx * right.forward, grad => grad * right.forward),
      'right -> GradientPort(right, dx => dx * left.forward, grad => grad * left.forward))
    override def toString = this.getClass.getName + (left, right)
  }

  class MultiplyByConstant(input: DifferentiableField, c: Float) extends DifferentiableField {
    override val inputs: Map[Symbol, GradientPort] = Map('input -> GradientPort(input, jacobian, jacobianAdjoint))
    override val batchSize: Int = input.batchSize
    override val forward: libcog.Field = input.forward * c

    def jacobian(dx: Field): Field = dx * c

    def jacobianAdjoint(grad: Field): Field = jacobian(grad)
    override def toString = this.getClass.getName + (input, c)
  }

  class DivideByConstant(input: DifferentiableField, c: Float) extends DifferentiableField {
    require(c != 0f, "cannot divide by zero")

    override val inputs: Map[Symbol, GradientPort] = Map('input -> GradientPort(input, jacobian, jacobianAdjoint))
    override val batchSize: Int = input.batchSize
    override val forward: libcog.Field = input.forward / c

    def jacobian(dx: Field): Field = dx / c

    def jacobianAdjoint(grad: Field): Field = jacobian(grad)
    override def toString = this.getClass.getName + (input, c)
  }

  class Pow(input: DifferentiableField, n: Float) extends DifferentiableField {
    private val in = (input.forward, input.batchSize)

    override val batchSize: Int = input.batchSize
    override val forward: Field = _forward(in)._1
    override val inputs: Map[Symbol, GradientPort] =
      Map('input -> GradientPort(input, jacobian(_, in), jacobian(_, in)))

    private def isIntPower = n == n.toInt

    private def _forward(in: (Field, Int)): (Field, Int) = {
      val (x, batchSize) = in
      if (isIntPower) {
        n.toInt match {
          case 1 => (x, batchSize)
          case 0 => throw new IllegalArgumentException("Pow node: power must be non-zero.")
          case intN => (libcog.pow(x, intN), batchSize)
        }
      }
      else
        (libcog.pow(x, n), batchSize)
    }

    private def jacobian(dx: Field, in: (Field, Int)): Field = {
      val (x, batchSize) = in
      if (isIntPower) {
        n.toInt match {
          case 1 => dx
          case 0 => throw new IllegalArgumentException("Pow node: power must be non-zero.")
          case intN => libcog.pow(x, intN - 1) * n * dx
        }
      }
      else
        libcog.pow(x, n - 1f) * n * dx
    }
    override def toString = this.getClass.getName + (input, n)
  }

}
