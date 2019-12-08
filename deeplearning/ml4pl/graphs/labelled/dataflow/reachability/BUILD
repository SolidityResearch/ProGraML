# Labelled graphs for reachability analysis.

py_library(
    name = "reachability",
    srcs = ["reachability.py"],
    visibility = ["//deeplearning/ml4pl/graphs/labelled/dataflow:__subpackages__"],
    deps = [
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/graphs/labelled/dataflow:data_flow_graphs",
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "reachability_test",
<<<<<<< HEAD:deeplearning/ml4pl/graphs/labelled/dataflow/reachability/BUILD
    size = "large",
=======
    size = "enormous",
>>>>>>> 767c784b1... Use size rather than timeout for describing tests.:deeplearning/ml4pl/graphs/labelled/reachability/BUILD
    srcs = ["reachability_test.py"],
    data = [
        "//deeplearning/ml4pl/testing/data:100_unlabelled_networkx_graphs",
    ],
    deps = [
        ":reachability",
        "//deeplearning/ml4pl/graphs:programl",
        "//deeplearning/ml4pl/graphs:programl_pb_py",
        "//deeplearning/ml4pl/testing:random_networkx_generator",
        "//deeplearning/ml4pl/testing:random_programl_generator",
        "//labm8/py:decorators",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)
