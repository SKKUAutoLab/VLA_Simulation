from setuptools import find_packages, setup

package_name = 'gui_pkg'

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
    description='PySide6 Mission Control GUI for autonomous vehicle simulation',
    entry_points={
        'console_scripts': [
            'mission_gui_node = gui_pkg.mission_gui_node:main',
            'debug_gui_node   = gui_pkg.debug_gui_node:main',
            'gt_extractor     = gui_pkg.gt_extractor:main',
            'parking_gt_extractor  = gui_pkg.parking_gt_extractor:main',
        ],
    },
)
