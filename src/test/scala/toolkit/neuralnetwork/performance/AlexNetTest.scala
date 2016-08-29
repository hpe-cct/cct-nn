package toolkit.neuralnetwork

import cogx.platform.opencl.OpenCLPlatform
import libcog._
import toolkit.neuralnetwork.examples.networks.AlexNet
import toolkit.neuralnetwork.operator.fourierProjectMAC

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** A regression test of AlexNet across batchSizes and available devices.
  *
  * Performance can often be increased by taking active control of clocks and fans speeds.
  *
  * The access to fan control is through nvidia-settings, which is tied to the /etc/X11/xorg.conf
  * file.  The GPUs permitting fan control must be listed in the xorg.conf file as driving
  * a screen and a monitor (if only a virtual one).  The following are some sample additions to xorg.conf
  * to permit fan control on an NVIDIA 1080 described as 'Device 0' that had no monitor attached:
  *
  * # Dummy monitor description
  * Section "Monitor"
  *   Identifier     "Monitor2"
  *   VendorName     "Unknown"
  *   ModelName      "CRT-0"
  *   HorizSync       0.0 - 0.0
  *   VertRefresh     0.0
  *   Option         "DPMS"
  * EndSection
  *
  * Section "Screen"
  *   Identifier     "Screen1"
  *   Device         "Device0"
  *   Monitor        "Monitor2"
  *   DefaultDepth    24
  *   Option         "ConnectedMonitor" "CRT"
  *   Option         "Coolbits" "12"
  *   Option         "nvidiaXineramaInfoOrder" "CRT-0"
  *   Option         "Stereo" "0"
  *   Option         "metamodes" "nvidia-auto-select +0+0"
  *   Option         "SLI" "Off"
  *   Option         "MultiGPU" "Off"
  *   Option         "BaseMosaic" "off"
  *   SubSection     "Display"
  *     Depth       24
  *   EndSubSection
  * EndSection
  *
  * Also, insert the following line into the xorg.conf "ServerLayout" section after the "Screen 0" line:
  *
  * Screen      1  "Screen1" RightOf "Screen0"
  *
  * Note the "Coolbits" line, a requirement for enabling fan control.  If your GPUs are already
  * driving a monitor, you can get by with only adding the "Coolbits" lines above.  Finally do an
  * internet search for nvidiafanspeed.py: a python script for controlling the fans.
  * We have a derivative version in-house that we haven't made externally available.
  *
  * By our experience, active fan control yields a modest 3-4% performance boost on the 1080, and a 5-8%
  * improvement on the TitanX.  For the TitanX only, an additional 2% boost was achieved
  * by actively setting clocks and power via (for a brick listed as GPU 1):
  *
  * nvidia-smi --persistence-mode=1
  * nvidia-smi -i 1 --auto-boost-default=0
  * nvidia-smi -i 1 --application-clocks=3505,1392
  * nvidia-smi -i 1 --power-limit=275
  *
  * Bottom line, you might get as much as 10% added performance on a TitanX by worrying about its configuration.
  *
  * We haven't played with clock offsets or over-voltages.  Those techniques, as well as anything mentioned here,
  * you are doing at your own risk.  Feel free to do your own research.
  *
  * @author Dick Carter
  */
object AlexNetTest extends App {

  // Some constants effecting the testing
  // When a device is first used, how long to run the model to potentially invoke thermal throttling before taking
  // measurements.  On subsequent tests, a lesser time can be used, since the GPU is only idle during the compile.
  val firstWarmUpSeconds = 120
  val notFirstWarmUpSeconds = 60
  // Time for actual measurements
  val testSeconds = 60
  // If instead you're interested in peak (pre-throttled) performance, you might want to spec some cool down time.
  val coolDownSeconds = 0
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

  /** Core routine that builds and times a single AlexNet model. */
  def timeAlexNet(batchSize: Int, device: Int, warmUpSeconds: Float, testSeconds: Float, coolDownSeconds: Float) = {
    val cg = new ComputeGraph(device = Some(device)) {
      new AlexNet(batchSize = batchSize, enableNormalization = true, useRandomData = false)
    }
    try {
      cg.reset
      if (warmUpSeconds > 0) {
        print(s"Beginning warm-up phase of $warmUpSeconds seconds...")
        cg.run
        Thread.sleep((warmUpSeconds*1000).toLong)
        cg.stop
        println("done.")
      }
      println(s"Beginning testing phase of $testSeconds seconds...")
      val testStartSimTick = cg.stop
      val start = System.nanoTime()
      cg.run
      Thread.sleep((testSeconds*1000).toLong)
      val testEndSimTick = cg.stop
      val durationMsec = (System.nanoTime() - start)/1000000.0
      val testSteps = testEndSimTick - testStartSimTick
      val stepTimeMsec =  durationMsec/testSteps
      val stepfreq =  1000.0/stepTimeMsec
      println(f"Step time = $stepTimeMsec%.3f msec. as measured over $testSteps steps (freq = $stepfreq%.3f Hz)")
      if (coolDownSeconds > 0) {
        print(s"Sleeping for $coolDownSeconds seconds to let the GPU cool...")
        Thread.sleep((coolDownSeconds*1000).toLong)
        println("done.\n")
      }
      stepTimeMsec
    }
    finally
      cg.release
  }

  val batchSizes = Seq(32, 64, 128, 256, 512)

  println(s"AlexNet regression over batchsizes $batchSizes and devices $deviceRange")

  // Loop over selected devices.  Then perform an inner test loop over the batchSizes.

  for (device <- deviceRange) {
    var warmUpSeconds = firstWarmUpSeconds
    for (batchSize <- batchSizes) {
      println(s"\n***************  Beginning batchSize $batchSize test " +
        s"on device $device (${deviceDescriptors(device)}) ******************\n")
      val stepTimeMsec = timeAlexNet(batchSize, device, warmUpSeconds, testSeconds, coolDownSeconds)
      val trainingRate = batchSize * (1000.0 / stepTimeMsec)
      println(f"\nFor batchsize ${batchSize}%d, training rate = ${trainingRate}%.2f images/sec.\n")
      warmUpSeconds = notFirstWarmUpSeconds
    }
  }

}
