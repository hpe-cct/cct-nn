package toolkit.neuralnetwork

import cogx.platform.opencl.OpenCLPlatform
import libcog._
import toolkit.neuralnetwork.examples.networks.AlexNet
import toolkit.neuralnetwork.operator.fourierProjectMAC

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** A regression test of AlexNet across batchSizes and available devices.
  *
  * @author Dick Carter
  */
object AlexNetTest extends App {

  // Some constants effecting the testing
  val warmUpSteps = 200
  val testSteps = 300
  val coolDownSeconds = 1
  val deviceNum = -1 // If -1 then all devices

  val deviceDescriptors = {
    val platform = OpenCLPlatform()
    try
      platform.devices.map(device => device.toString)
    finally
      platform.release()
  }
  def numDevices = deviceDescriptors.length

  val deviceRange =
    if (deviceNum == -1)
      0 to numDevices - 1
    else
      deviceNum to deviceNum

  /** Core routine that builds and times a single FourierProjectMAC kernel instance. */
  def timeAlexNet(batchSize: Int, device: Int) = {
    val cg = new ComputeGraph(device = Some(device)) {
      new AlexNet(batchSize = batchSize, enableNormalization = true, useRandomData = false)
    }
    cg.reset
    println("Beginning warm-up phase.")
    cg.step(warmUpSteps)
    println("Beginning testing phase.")
    val start = System.nanoTime()
    cg.step(testSteps)
    val durationMsec = (System.nanoTime() - start)/1000000.0
    val stepTimeMsec =  durationMsec/testSteps
    val stepfreq =  1000.0/stepTimeMsec
    println(f"Step time = $stepTimeMsec%.3f msec. (freq = $stepfreq%.3f Hz)")
    cg.release
    stepTimeMsec
  }

  val batchSizes = Seq(32, 32, 64, 128, 256, 512)

  println(s"AlexNet regression over batchsizes $batchSizes and devices $deviceRange")

  // Loop over selected devices.  Then perform an inner test loop over the batchSizes.

  for (device <- deviceRange) {
    for (batchSize <- batchSizes) {
      println(s"\n***************  Beginning testing of batchSize $batchSize " +
        s"on device $device (${deviceDescriptors(device)}) ******************\n")

      val stepTimeMsec = timeAlexNet(batchSize, device)
      val trainingRate = batchSize * (1000.0 / stepTimeMsec)
      println(s"\nFor batchsize $batchSize, training rate = $trainingRate images/sec.\n")
      print(s"Sleeping for $coolDownSeconds sec to let the GPU cool...")
      Thread.sleep(coolDownSeconds * 1000)
      println("done.\n")
    }
  }

}
