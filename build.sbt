name := "cct-nn"

description := "CCT neural network library."

organizationName := "Hewlett Packard Labs"

organizationHomepage := Some(url("http://www.labs.hpe.com"))

version := "2.0.0-alpha.4"

organization := "com.hpe.cct"

scalaVersion := "2.11.7"

parallelExecution in Test := false

fork in run := true

fork in Test := true

javaOptions in Test ++= Seq("-Xmx100G", "-Xloggc:gc.log")

libraryDependencies ++= Seq(
  "com.hpe.cct" %% "cct-core" % "5.0.0",
  "com.hpe.cct" %% "cct-io" % "0.8.9",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "junit" % "junit" % "4.7" % "test",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0" % "test",
  "ch.qos.logback" %  "logback-classic" % "1.1.7" % "test"
)

licenses += ("Apache-2.0", url("https://www.apache.org/licenses/LICENSE-2.0.html"))

resolvers += Resolver.jcenterRepo

bintrayRepository := "maven"

bintrayOrganization := Some("hpe-cct")
