#!/bin/bash

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package &&
sleep 1 &&
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/freetf2_test/tensorflow_pkg &&
sleep 1 &&
python -m pip install --upgrade /tmp/freetf2_test/tensorflow_pkg/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl
