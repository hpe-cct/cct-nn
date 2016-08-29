package toolkit.neuralnetwork

import cogx.platform.opencl.OpenCLPlatform
import libcog._
import toolkit.neuralnetwork.operator.fourierProjectMAC

import scala.collection.mutable

/** A regression test of the FourierProjectMAC GPUOperator across two thread-coarsening parameters.
  * Editable constants dictate which devices are tested and which AlexNet layers govern the field sizes.
  *
  * @author Dick Carter
  */
object FourierProjectMACTest extends App {

  // Some constants effecting the testing
  val warmUpSteps = 50
  val testSteps = 300
  val coolDownSeconds = 20
  val batchSize = 256
  val deviceNum = -1 // If -1 then all devices
  val layerNum = -1  // If -1, then AlexNet layer 2, 3 and 4.

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
  val layerRange =
    if (layerNum == -1)
      2 to 4
    else
      layerNum to layerNum

  /** Core routine that builds and times a single FourierProjectMAC kernel instance. */
  def timeProjectMAC(layer: LayerParams, batchSize: Int, batchSetSize: Int, filterSetSize: Int, device: Int) = {
    val cg = new ComputeGraph(device = Some(device)) {

      val tensorSize1 = layer.inputFeatures * batchSize
      val tensorSize2 = layer.inputFeatures * layer.outputFeatures

      val columns = Logarithm.roundUpPowerOf2(layer.inputSize)
      val rows = columns / 2 + 1

      val fieldShape = Shape(rows, columns)
      val tensorShape1 = Shape(tensorSize1)
      val tensorShape2 = Shape(tensorSize2)
      val in1real = VectorField(fieldShape, tensorShape1)
      val in1imaginary = VectorField(fieldShape, tensorShape1)
      val in2real = VectorField(fieldShape, tensorShape2)
      val in2imaginary = VectorField(fieldShape, tensorShape2)
      val (outreal, outimaginary) = fourierProjectMAC((in1real, in1imaginary), (in2real, in2imaginary),
        batchSize, batchSetSize, filterSetSize)
      probe(outreal, outimaginary)
    }
    cg.reset
    cg.step(warmUpSteps)
    val start = System.nanoTime()
    cg.step(testSteps)
    val durationMsec = (System.nanoTime() - start)/1000000.0
    val stepTimeMsec =  durationMsec/testSteps
    val stepfreq =  1000.0/stepTimeMsec
    println(f"Step time = $stepTimeMsec%.3f msec. (freq = $stepfreq%.3f Hz)")
    cg.release
    stepTimeMsec
  }

  // Note that any downsampling is applied by a different kernel than the FourierProjectMAC, so the
  // input size == output size.  The outputFeatures is essentially the number of logical filters.
  case class LayerParams(name: String, inputFeatures: Int, outputFeatures: Int, inputSize: Int)

  /** AlexNet layer sizes. */
  def alexnetLayer(i: Int): LayerParams = {
    i match {
      case 1 => throw new RuntimeException("AlexNet layer 1 has strided convolution and doesn't use FourierProjectMAC.")
      case 2 => new LayerParams("AlexNet Layer 2", 96, 256, 55)
      case 3 => new LayerParams("AlexNet Layer 3", 256, 384, 27)
      case 4 => new LayerParams("AlexNet Layer 4", 384, 384, 13)
      case x => throw new RuntimeException(s"AlexNet layer $x doesn't exist or is fully connected.")
    }
  }

  case class TestParam(batchSetSize: Int, filterSetSize: Int)

  println(s"FourierProjectMAC regression over tuning parameters for AlexNet layers $layerRange and devices $deviceRange")

  // Loop over selected devices and layers.  Then perform an inner test loop over the kernel tuning parameters.

  for (device <- deviceRange) {
    for (layer <- layerRange) {
      println(s"\n***************  Beginning testing of AlexNet layer $layer " +
        s"on device $device (${deviceDescriptors(device)}) ******************\n")
      val results = mutable.HashMap[TestParam, Double]()

      val testCases =
        for (batchSetSize <- 2 to 12 by 2; filterSetSize <- 2 to 12 by 2)
          yield (TestParam(batchSetSize, filterSetSize))

      for (testCase <- testCases) {
        val batchSetSize = testCase.batchSetSize
        val filterSetSize = testCase.filterSetSize
        println(s"Starting test with (batchSetSize,filterSetSize) = ($batchSetSize,$filterSetSize)")
        val layerParam = alexnetLayer(layer)
        val newTime = timeProjectMAC(layerParam, batchSize, batchSetSize, filterSetSize, device)
        results.put(testCase, newTime)
        print(s"Sleeping for $coolDownSeconds sec to let the GPU cool...")
        Thread.sleep(coolDownSeconds * 1000)
        println("done.\n")
      }
      // Find the result Map entry with the lowest step value (the second parameter of the key-value pair)
      val (bestParams, bestStepTime) = results.minBy(_._2)
      val bestFreq = 1000.0 / bestStepTime
      print(s"Best (batchSetSize, filterSetSize) = (${bestParams.batchSetSize},${bestParams.filterSetSize}).  ")
      println(s"Step time  = $bestStepTime msec, freq = $bestFreq.")

      println(s"\nSummary for AlexNet layer $layer on device $device (${deviceDescriptors(device)}):\n")
      println("Params\tFreq\n")
      for (testCase <- testCases) {
        val batchSetSize = testCase.batchSetSize
        val filterSetSize = testCase.filterSetSize
        val freq = 1000.0 / results(testCase)
        println(f"($batchSetSize%d,$filterSetSize%d)\t$freq%5.2f")
      }
    }
  }

}
