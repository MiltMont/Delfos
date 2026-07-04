nanobind_add_module(_delfos NB_DOMAIN delfos STABLE_ABI py_delfos.cpp)
target_link_libraries(_delfos PRIVATE delfos_core)

# When installed as part of the wheel, the module lives inside the delfos
# package so it is importable as `from delfos._delfos import Store`.
install(TARGETS _delfos DESTINATION delfos)
