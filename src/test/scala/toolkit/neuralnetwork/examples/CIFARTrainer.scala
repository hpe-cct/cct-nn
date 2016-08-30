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

package toolkit.neuralnetwork.examples

import cogio.FieldState
import com.typesafe.scalalogging.StrictLogging
import libcog._
import toolkit.neuralnetwork.WeightStore
import toolkit.neuralnetwork.examples.networks.Net


object CIFARTrainer extends App with StrictLogging {
  val batchSize = 100
  val netName = 'SimpleConvNet

  def validate(snapshot: Map[Symbol, FieldState]): (Float, Float) = {
    val cg = new ComputeGraph {
      val net = Net(netName, useRandomData = false, learningEnabled = false, batchSize = batchSize,
        training = false, weights = WeightStore.restoreFromSnapshot(snapshot))

      probe(net.correct)
      probe(net.loss.forward)

      def readLoss(): Float = {
        read(net.loss.forward).asInstanceOf[ScalarFieldReader].read()
      }

      def readCorrect(): Float = {
        read(net.correct).asInstanceOf[ScalarFieldReader].read()
      }
    }

    val steps = 10000 / batchSize
    var lossAcc = 0f
    var correctAcc = 0f

    cg withRelease {
      cg.reset

      for (i <- 0 until steps) {
        lossAcc += cg.readLoss()
        correctAcc += cg.readCorrect()

        cg.step
      }
    }

    (lossAcc / steps, correctAcc / steps)
  }

  val cg = new ComputeGraph {
    val net = Net(netName, useRandomData = false, learningEnabled = true, batchSize = batchSize)

    probe(net.correct)
    probe(net.loss.forward)

    def readLoss(): Float = {
      read(net.loss.forward).asInstanceOf[ScalarFieldReader].read()
    }

    def readCorrect(): Float = {
      read(net.correct).asInstanceOf[ScalarFieldReader].read()
    }
  }

  cg withRelease {
    logger.info(s"starting compilation")
    cg.reset
    logger.info(s"compilation finished")

    val loss = cg.readLoss()
    val correct = cg.readCorrect()

    logger.info(s"initial loss: $loss")
    logger.info(s"initial accuracy: $correct")

    logger.info(s"Iteration: 0 Sample: 0 Training Loss: $loss Training Accuracy: $correct")

    for (i <- 1 to 50000) {
      cg.step

      if (i % 100 == 0) {
        val loss = cg.readLoss()
        val correct = cg.readCorrect()
        logger.info(s"Iteration: $i Sample: ${i * batchSize} Training Loss: $loss Training Accuracy: $correct")
      }

      if (i % 500 == 0) {
        logger.info(s"Validating...")
        val (loss, correct) = validate(cg.net.weights.snapshot(cg))
        logger.info(s"Iteration: $i Sample: ${i * batchSize} Validation Loss: $loss Validation Accuracy: $correct")
      }
    }
  }
}
