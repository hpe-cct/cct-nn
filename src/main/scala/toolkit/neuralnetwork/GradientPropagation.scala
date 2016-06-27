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
import toolkit.neuralnetwork.DifferentiableField.{GradientBinding, Leaf, Sender}

import scala.collection.mutable


trait GradientPropagation { self: DifferentiableField =>
  def activateSGD(initField: Field = ScalarField(1f), invokeCallbacks: Boolean = true): Unit = {
    require(forward.fieldType == initField.fieldType,
      s"forward fieldType (${forward.fieldType}) must match initField fieldType (${initField.fieldType})")
    require(gradientBinding.isEmpty, "SGD is already active on this differentiable field")

    val leafNodes = mutable.Set[DifferentiableField]()

    /// STEP A2: Search back from the loss node to classify each node as a Leaf, Sender, or Unused
    def checkGradients(df: DifferentiableField, sender: Option[(DifferentiableField, Symbol)], owner: DifferentiableField): Unit = {
      if (df.gradientBinding.isDefined) {
        require(df.gradientBinding.get.owner == owner, "cannot activate SGD, encountered a conflicting root field")

        (df.gradientBinding.get.gradientType, sender) match {
          case (Leaf(set), Some(sn)) => df.gradientBinding = Some(GradientBinding(owner, Leaf(set + sn)))
          case (Sender(set), Some(sn)) => df.gradientBinding = Some(GradientBinding(owner, Sender(set + sn)))
          case _ => //pass
        }

        return
      }

      // Search down the DAG, passing along the symbol for this node and the input index
      df.inputs.foreach(in => checkGradients(in._2.df, Some(df, in._1), owner))

      def isSender(df: DifferentiableField): Boolean = {
        df.inputs.exists(p => {
          val gt = p._2.df.gradientBinding
          gt.exists(g => g.gradientType.isInstanceOf[Leaf] || g.gradientType.isInstanceOf[Sender])
        })
      }

      if (df.gradientConsumer) {
        require(df.inputs.isEmpty, "state nodes may not have input ports")
        leafNodes += df
        df.gradientBinding = Some(GradientBinding(owner, Leaf(sender.toSet)))
      } else if (isSender(df)) {
        df.gradientBinding = Some(GradientBinding(owner, Sender(sender.toSet)))
      }
    }

    // Run the gradient pass starting from the loss node
    checkGradients(this, None, this)

    if (leafNodes.isEmpty) {
      // No nodes actually need a gradient. Aborting.
      return
    }

    // Initialize the backwards field of the loss node to 1
    backward = Some(initField)

    if (invokeCallbacks) {
      backwardCallback(backward.get)
    }

    /// STEP B1: Wire the gradient through the DAG from each leaf node back to the loss node
    def wireGradients(df: DifferentiableField): Unit = {
      if (df.backward.isDefined) {
        return
      }

      val senders = df.gradientBinding.get.gradientType match {
        case Leaf(set) => set
        case Sender(set) => set
        case _ => throw new RuntimeException("tried to wire the gradient through an unused node")
      }

      senders.foreach(s => wireGradients(s._1))
      assert(senders.forall(s => s._1.backward.isDefined))

      val gradients = senders.map(s => {
        val sender = s._1
        val senderField = sender.backward.get
        // Make sure the sender has an appropriate input port
        require(sender.inputs.contains(s._2))
        // Pass the upstream backwards field through the appropriate gradient port
        val backField = sender.inputs(s._2).jacobianAdjoint(senderField)
        // Make sure the backwards type is consistent
        require(backField.fieldType == df.forward.fieldType, s"expected type ${df.forward.fieldType}, got ${backField.fieldType} from port ${s._2} on $sender")
        backField
      })

      df.backward = Some(gradients.reduce(_ + _))

      if (invokeCallbacks) {
        df.backwardCallback(df.backward.get)
      }
    }

    // Run the wireGradient process rooted at *each* leaf node
    leafNodes.foreach(df => wireGradients(df))

    // All leaf nodes should have their backward field defined now
    assert(leafNodes.forall(df => df.backward.isDefined))
  }

  private def _totalDerivative(df: DifferentiableField): Field = {
    if (df.forwardGradient.isDefined) {
      return df.forwardGradient.get
    }

    val fg = (df.gradientConsumer, df.inputs) match {
      // No partial derivatives available. This is a static DifferentiableField.
      case (false, in) if in.isEmpty => Field(df.forward.fieldType)

      // This is a source of independent variables. Pass them along.
      case (true, in) if in.isEmpty => Field(df.forward.fieldType) + 1f

      // Ordinary intermediate node. Recurse up the graph.
      case (false, in) =>
        df.inputs.map(i => {
          // Get the total derivative for the upstream DifferentiableField, then pass it through
          // the appropriate Jacobian function on this DifferentiableField
          val fDx = _totalDerivative(i._2.df)
          i._2.jacobian(fDx)
        }).reduce(_ + _)

      case _ => throw new RuntimeException("DifferentialField cannot both consume a gradient and have inputs")
    }

    df.forwardGradient = Some(fg)
    fg
  }

  def totalDerivative(): Field = {
    _totalDerivative(this)
  }
}
