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

package toolkit.neuralnetwork.source

import java.io._

import scala.util.Try

private [source] object StreamLoader {
  def apply(path: String, resourcePath: String) = new BufferedInputStream({
    val stream = Try(new FileInputStream(path)) recover {
      case e: FileNotFoundException => new FileInputStream(resourcePath + path)
    } recover {
      case e: FileNotFoundException =>
        assert(this.getClass.getClassLoader.getResource(path) != null)
        this.getClass.getClassLoader.getResourceAsStream(path)
    }

    assert(stream.isSuccess, s"failed to find resource '$path' with resource path '$resourcePath'")

    stream.get
  })

  def length(path: String, resourcePath: String, estimate: Option[Long]): Long = {
    if (new File(path).exists()) {
      new File(path).length()
    } else if (new File(resourcePath + path).exists()) {
      new File(resourcePath + path).length()
    } else {
      assert(this.getClass.getClassLoader.getResource(path) != null, s"failed to find resource '$path' with resource path '$resourcePath'")
      assert(estimate.isDefined, s"must supply a length estimate to load resource '$path' from a jar")
      estimate.get
    }
  }
}
