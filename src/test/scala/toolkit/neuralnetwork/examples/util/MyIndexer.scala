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

package toolkit.neuralnetwork.examples.util

/**
 * Created by symonsj on 3/29/16.
 *
 * This is used for providing an index from 0..9 for the AlexNet validation step.
 *
 */


object MyIndexer {
  var count = -1

  def nextIdx: Int = {
    count += 1
    val idx = count % 50    // should be 10, hack for now at 50
    idx
  }
}



