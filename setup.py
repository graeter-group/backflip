from setuptools import setup

setup(
    name="backflip",
    packages=[
        "backflip",
    ],
    package_dir={
        "backflip": './backflip',
    },
    entry_points={
        "console_scripts": [
            "backflip-predict=backflip.deployment.cmd_line_entry:backflip_predict_cli",
            "backflip-annotate=backflip.deployment.cmd_line_entry:backflip_annotate_cli",
        ],
    },
)
