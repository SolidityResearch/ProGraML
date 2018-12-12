# Package containing Jupyter notebook servers.

# A Jupyter notebook server that includes a bunch of dependencies which I
# find useful. If you would like to include extra dependencies, create a new
# py_binary target with this target as the srcs, and add the dependencies:
#
#    py_binary(
#      name = "notebook",
#      srcs = ["//notebooks"],
#      deps = [
#          # Your dependencies here.
#      ],
#    )
#
<<<<<<< HEAD
# Use the run_notebooks.sh script to launch this target in the source tree:
# bash path/to/package/run_notebooks.sh.
=======
# Use the run_notebooks.sh script to launch this target in the root of the
# source tree: bash notebooks/run_notebooks.sh.
>>>>>>> 77bad6326... Move //labm8:notebooks to //notebooks.
py_binary(
    name = "notebooks",
    srcs = ["notebooks.py"],
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:jsonutil",
        "//labm8/py:labdate",
        "//labm8/py:labtypes",
        "//labm8/py:logutil",
        "//labm8/py:pbutil",
        "//labm8/py:pdutil",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
        "//labm8/py:viz",
        "//third_party/py/bokeh",
        "//third_party/py/fs",
=======
        "//labm8:bazelutil",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//labm8:viz",
        "//third_party/py/absl",
        "//third_party/py/bokeh",
>>>>>>> 77bad6326... Move //labm8:notebooks to //notebooks.
        "//third_party/py/graph_nets",
        "//third_party/py/jupyter",
        "//third_party/py/keras",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
<<<<<<< HEAD
        "//third_party/py/scikit_learn",
        "//third_party/py/scipy",
=======
>>>>>>> 77bad6326... Move //labm8:notebooks to //notebooks.
        "//third_party/py/seaborn",
        "//third_party/py/sonnet",
        "//third_party/py/tensorflow",
    ],
)
