# SPDX-License-Identifier: CC0-1.0

camhal_dep = dependency('libcamhal', required : false)

if camhal_dep.found()
    libcamera_internal_sources += files([
        'ipu6.cpp',
    ])
    libcamera_deps += [camhal_dep]
else
    warning('libcamhal not found, IPU6 pipeline handler will not be built')
    # Remove ipu6 from pipelines list so the build doesn't fail
    pipeline = ''
endif
