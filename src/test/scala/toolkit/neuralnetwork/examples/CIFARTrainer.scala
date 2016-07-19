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

import com.typesafe.scalalogging.StrictLogging
import libcog._
import toolkit.neuralnetwork.examples.networks.CIFAR


object CIFARTrainer extends App with StrictLogging {
  val batchSize = 100

  val cg1 = new ComputeGraph {
    val net = new CIFAR(useRandomData = false, learningEnabled = true, batchSize = batchSize)
    probe(net.correct)
  }

  def readLoss(): Float = {
    cg1.read(cg1.net.loss.forward).asInstanceOf[ScalarFieldReader].read()
  }

  def readCorrect(): Float = {
    cg1.read(cg1.net.correct).asInstanceOf[ScalarFieldReader].read()
  }

  cg1 withRelease {
    logger.info(s"starting compilation")
    cg1.reset
    logger.info(s"compilation finished")

    val loss = readLoss()
    val correct = readCorrect()

    logger.info(s"initial loss: $loss")
    logger.info(s"initial accuracy: $correct")

    for (i <- 1 to 50000) {
      cg1.step

      if (i%100 == 0) {
        val loss = readLoss()
        val correct = readCorrect()
        logger.info(s"Iteration: $i Sample: ${i * batchSize} Loss: $loss Accuracy: $correct")
      }
    }
  }
}
