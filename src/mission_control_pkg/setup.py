from setuptools import find_packages, setup

package_name = 'mission_control_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='autolab',
    description='Mission task manager with sequential execution and lane change avoidance',
    entry_points={
        'console_scripts': [
            'task_manager_node = mission_control_pkg.task_manager_node:main',
        ],
    },
)
