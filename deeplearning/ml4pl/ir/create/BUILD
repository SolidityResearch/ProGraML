# Datasets for compiler CFG problems.

py_binary(
    name = "import_from_directory",
    srcs = ["import_from_directory.py"],
    deps = [
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
=======
        "//deeplearning/ml4pl/bytecode:bytecode_database",
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//labm8:app",
        "//labm8:fs",
        "//labm8:sqlutil",
>>>>>>> a6aea9070... Add a script to import bytecode from a database.:deeplearning/ml4pl/bytecode/create/BUILD
=======
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/bytecode/create/BUILD
    ],
)

py_binary(
    name = "import_from_github",
    srcs = ["import_from_github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//deeplearning/ml4pl/ir:ir_database",
=======
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/bytecode/create/BUILD
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:lockfile",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)

py_library(
    name = "import_from_linux",
    srcs = ["import_from_linux.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//datasets/linux",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//deeplearning/ml4pl/ir:ir_database",
=======
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/bytecode/create/BUILD
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/pyparsing",
    ],
)

<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
# TODO(github.com/ChrisCummins/ProGraML/issues/7): Linux build is broken.
=======
# TODO(cec): Broken.
>>>>>>> 6c5b8358f... Update tests.:deeplearning/ml4pl/bytecode/create/BUILD
=======
# TODO(cec): Linux build is broken. Investigate later.
>>>>>>> 8d355fc4c... Remove unused BUILD target.:deeplearning/ml4pl/bytecode/create/BUILD
=======
# TODO(github.com/ChrisCummins/ProGraML/issues/7): Linux build is broken.
>>>>>>> d9a9c98fc... Add cross-references to issue tracker.:deeplearning/ml4pl/bytecode/create/BUILD
#py_test(
#    name = "import_from_linux_test",
#    srcs = ["import_from_linux_test.py"],
#    deps = [
#        ":import_from_linux",
#        "//datasets/linux",
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
#        "//labm8/py:app",
#        "//labm8/py:system",
#        "//labm8/py:test",
#        "//third_party/py/pytest",
#    ],
#)

py_binary(
    name = "import_from_npb",
    srcs = ["import_from_npb.py"],
    deps = [
        "//compilers/llvm:llvm_dis",
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
    ],
)
=======
#        "//labm8:app",
#        "//labm8:system",
#        "//labm8:test",
=======
#        "//labm8/py:app",
#        "//labm8/py:system",
#        "//labm8/py:test",
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/bytecode/create/BUILD
#        "//third_party/py/pytest",
#    ],
#)
>>>>>>> 6c5b8358f... Update tests.:deeplearning/ml4pl/bytecode/create/BUILD

py_binary(
    name = "import_from_npb",
    srcs = ["import_from_npb.py"],
    deps = [
        "//compilers/llvm:llvm_dis",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
    ],
)

py_library(
    name = "import_from_pact17_devmap",
    srcs = ["import_from_pact17_devmap.py"],
    deps = [
        "//compilers/llvm:clang",
        "//compilers/llvm:opt_util",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//deeplearning/ml4pl/ir:ir_database",
=======
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/ml4pl/bytecode/create/BUILD
        "//labm8/py:app",
        "//labm8/py:decorators",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
    ],
)

py_test(
    name = "import_from_pact17_devmap_test",
    srcs = ["import_from_pact17_devmap_test.py"],
    deps = [
        ":import_from_pact17_devmap",
        "//labm8/py:app",
        "//labm8/py:system",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "import_from_poj104",
    srcs = ["import_from_poj104.py"],
    deps = [
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD
        "//deeplearning/ml4pl/ir:ir_database",
        "//labm8/py:app",
        "//labm8/py:fs",
=======
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//labm8/py:app",
        "//labm8/py:fs",
    ],
)
<<<<<<< HEAD:deeplearning/ml4pl/ir/create/BUILD

py_binary(
    name = "populate_cfg_table",
    srcs = ["populate_cfg_table.py"],
    deps = [
        "//compilers/llvm:opt",
        "//compilers/llvm:opt_util",
        "//deeplearning/ml4pl:ml4pl_pb_py",
        "//deeplearning/ml4pl/bytecode:bytecode_database",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph",
        "//deeplearning/ml4pl/graphs/unlabelled/cfg:llvm_util",
        "//labm8:app",
>>>>>>> a811cbd12... Auto-format BUILD files.:deeplearning/ml4pl/bytecode/create/BUILD
    ],
)
=======
>>>>>>> 8d355fc4c... Remove unused BUILD target.:deeplearning/ml4pl/bytecode/create/BUILD
