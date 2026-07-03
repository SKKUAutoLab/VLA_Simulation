from setuptools import find_packages, setup

package_name = 'qwen_vl_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='autolab',
    description='Qwen3-VL end-to-end autonomous driving node',
    entry_points={
        'console_scripts': [
            'qwen_vl_driver_node = qwen_vl_pkg.qwen_vl_driver_node:main',
            'qwen_vl_trt_node    = qwen_vl_pkg.qwen_vl_trt_node:main',
            'vla_brain_node      = qwen_vl_pkg.vla_brain_node:main',
            'vla_agent_node      = qwen_vl_pkg.vla_agent_node:main',
            'vla_cmd_node        = qwen_vl_pkg.vla_cmd_node:main',
        ],
    },
)
