# A wrapper around pip package to pull in undeclared dependencies.

load("@requirements//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GPLv2

py_library(
    name = "pyfiglet",
    srcs = ["//third_party/py:empty.py"],
    deps = [
<<<<<<< HEAD:third_party/py/pyfiglet/BUILD
        requirement("pyfiglet"),
=======
        requirement("bibtexparser"),
        "//third_party/py/pyparsing",
>>>>>>> 3fe6917f5... Add missing implicit dependency.:third_party/py/bibtexparser/BUILD
    ],
)
