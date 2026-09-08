from setuptools import find_packages, setup

package_name = "robe_stretch"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/execute_pull.launch.py",
            "launch/workstation_executor.launch.py",
            "launch/sample_origin.launch.py",
            "launch/preview_origin.launch.py",
        ]),
        ("share/" + package_name + "/config", [
            "config/executor.yaml",
            "config/grasp_tune.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    tests_require=[],
    zip_safe=True,
    maintainer="RoBE maintainers",
    maintainer_email="rchi-lab@example.com",
    description="Policy-faithful Stretch BedPull executor",
    license="MIT",
    entry_points={
        "console_scripts": [
            "action_executor = robe_stretch.action_executor:main",
            "bed_origin_node = robe_stretch.bed_origin_node:main",
            "send_bed_pull = robe_stretch.send_bed_pull:main",
            "tune_grasp = robe_stretch.tune_grasp:main",
            "preflight_bed_pull = robe_stretch.preflight:main",
            "health_check = robe_stretch.health_check:main",
            "sample_origin = robe_stretch.sample_origin:main",
            "preview_origin = robe_stretch.preview_origin:main",
        ],
    },
)
