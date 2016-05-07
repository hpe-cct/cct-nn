name := "cogx-nn"

description := "CogX neural network library."

organizationName := "Hewlett Packard Labs"

organizationHomepage := Some(url("http://www.labs.hpe.com"))

version := "2.0.0-alpha.1"

organization := "com.hpe.cct"

scalaVersion := "2.11.7"

parallelExecution in Test := false

fork in run := true

fork in Test := true

javaOptions in Test ++= Seq("-Xmx100G", "-Xloggc:gc.log")

libraryDependencies ++= Seq(
  "com.hpe.cct" %% "cogx" % "4.4.9",
  "com.hpe.cct" %% "cogx-io" % "0.8.6",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "junit" % "junit" % "4.7" % "test"
)

licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0.html"))

bintrayRepository := "maven"

bintrayOrganization := Some("cogexmachina")
