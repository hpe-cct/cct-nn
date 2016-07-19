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

package toolkit.neuralnetwork.performance

import com.typesafe.scalalogging.StrictLogging
import toolkit.neuralnetwork.examples.networks.CIFAR

import scala.collection.mutable.ListBuffer
import libcog._


object Benchmark extends App with StrictLogging {
  val (net, batchSize) = args.length match {
    case 0 => ("cifar10_quick", 256)
    case 1 => (args(0), 256)
    case 2 => (args(0), args(1).toInt)
    case _ => throw new RuntimeException(s"illegal arguments (${args.toList})")
  }

  require(net == "cifar10_quick", s"network $net isn't supported")

  logger.info(s"net: $net")
  logger.info(s"batch size: $batchSize")

  val cg1 = new ComputeGraph {
    val net = new CIFAR(useRandomData = true, learningEnabled = false, batchSize = batchSize)
  }

  val forward = new ListBuffer[Double]()
  val backward = new ListBuffer[Double]()

  cg1 withRelease {
    logger.info(s"starting compilation (inference)")
    cg1.step
    logger.info(s"compilation finished (inference)")

    for (i <- 1 to 50) {
      val start = System.nanoTime()
      cg1.step
      val stop = System.nanoTime()
      val elapsed = (stop - start).toDouble / 1e6
      logger.info(s"Iteration: $i forward time: $elapsed ms.")
      forward += elapsed
    }
  }

  val cg2 = new ComputeGraph {
    val net = new CIFAR(useRandomData = true, learningEnabled = true, batchSize = batchSize)
  }

  cg2 withRelease {
    logger.info(s"starting compilation (learning)")
    cg2.step
    logger.info(s"compilation finished (learning)")

    for (i <- 1 to 50) {
      val start = System.nanoTime()
      cg2.step
      val stop = System.nanoTime()
      val elapsed = (stop - start).toDouble / 1e6
      logger.info(s"Iteration: $i forward-backward time: $elapsed ms.")
      backward += elapsed
    }
  }

  logger.info(s"Average Forward pass: ${forward.sum / forward.length} ms.")
  logger.info(s"Average Forward-Backward: ${backward.sum / backward.length} ms.")
}
