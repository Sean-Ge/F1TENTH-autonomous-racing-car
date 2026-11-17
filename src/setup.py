from glob import glob
from setuptools import find_packages, setup

package_name = 'milestone3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahnaf-rashid',
    maintainer_email='ahnaf-rashid@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "safety_node=milestone3.safety_node:main",
            "wall_follow_node=milestone3.wall_follow_node:main",
            "master_control_node=milestone3.master_control_node:main",
            "gap_follow_node=milestone3.gap_follow_node:main",
            "straight_drive_node=milestone3.straight_drive_node:main",
            "vision_safety_node=milestone3.vision_safety_node:main",
            "vision_follow_node=milestone3.vision_follow_node:main",
            "lap_counter_node=milestone3.laps_counting_node:main",
        ],
    },
)
