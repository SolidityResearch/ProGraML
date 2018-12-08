# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

py_library(
    name = "bokeh",
<<<<<<< HEAD
    srcs = ["//third_party/py:empty.py"],
=======
    srcs = ["bokeh.py"],
>>>>>>> 31e381197... Add new third party python packages.
    deps = [
        requirement("bokeh"),
        requirement("pandas-bokeh"),
    ],
)
